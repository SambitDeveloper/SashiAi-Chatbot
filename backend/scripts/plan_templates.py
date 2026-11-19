"""Administer the semantic query-plan template cache (app/engine/semantic_cache.py).

Runs with the cache's OWN credential (PLAN_CACHE_MONGODB_URI, readWrite on PLAN_CACHE_DATABASE only) — never with
MONGODB_URI. Templates hold only slot-replaced question text and plan structure: no names, ids or results.

    uv run python -m scripts.plan_templates verify                      # connection, collection, vector index status
    uv run python -m scripts.plan_templates create-index --admin-uri-env ATLAS_ADMIN_URI
                                                                        # one-off, needs a user allowed to create
                                                                        # Atlas Search indexes (NOT the app user)
    uv run python -m scripts.plan_templates update-index --admin-uri-env ATLAS_ADMIN_URI   # apply a changed definition
    uv run python -m scripts.plan_templates list [--revoked]
    uv run python -m scripts.plan_templates show <template_id>
    uv run python -m scripts.plan_templates revoke <template_id> --by "<name>" --reason "<why>"

Templates are reusable as soon as they are stored (no review step); every reuse is re-validated against the new
question. If a template turns out wrong, `revoke` switches it off on the next query; it is never re-enabled by being
stored again.
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from app.config import get_settings

EMBEDDING_DIMENSIONS = 384  # BAAI/bge-small-en-v1.5 (verified: fastembed output length)


def index_definition() -> dict:
    """Atlas Vector Search index: cosine over the 384-d BGE embedding, with the fields the query pre-filters on."""
    return {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": EMBEDDING_DIMENSIONS, "similarity": "cosine"},
            {"type": "filter", "path": "revoked"},
            {"type": "filter", "path": "compat_key"},
        ]
    }


def _collection(uri: str | None = None):
    s = get_settings()
    uri = uri or s.plan_cache_mongodb_uri
    if not uri:
        sys.exit("PLAN_CACHE_MONGODB_URI is not set")
    if s.mongodb_uri and uri.strip() == s.mongodb_uri.strip():
        sys.exit("refusing to use MONGODB_URI (the resident credential) for the template cache")
    client = MongoClient(uri, serverSelectionTimeoutMS=15000, appname="shashicare-plan-templates-admin")
    return client, client[s.plan_cache_database][s.plan_cache_collection], s


def _check_privileges(client, s) -> None:
    """The cache user must be restricted to the cache database. Connection strings can differ in text while logging in
    as the same broad user, so this checks the authenticated user and its roles (names are never printed)."""
    auth = client.admin.command("connectionStatus")["authInfo"]
    roles = [(r["role"], r["db"]) for r in auth["authenticatedUserRoles"]]
    problems = []
    if s.mongodb_uri:
        other = MongoClient(s.mongodb_uri, serverSelectionTimeoutMS=15000).admin.command("connectionStatus")["authInfo"]
        if [u["user"] for u in other["authenticatedUsers"]] == [u["user"] for u in auth["authenticatedUsers"]]:
            problems.append("logs in as the SAME database user as MONGODB_URI")
    broad = [r for r, db in roles if r.endswith("AnyDatabase") or r in ("root", "atlasAdmin", "dbAdminAnyDatabase")]
    if broad:
        problems.append(f"has cluster-wide roles {broad}")
    outside = [(r, db) for r, db in roles if db not in (s.plan_cache_database, "admin") or (db == "admin" and r not in broad)]
    if outside:
        problems.append(f"has roles outside {s.plan_cache_database}: {outside}")
    print("cache user roles:", roles)
    if problems:
        print("NOT RESTRICTED — " + "; ".join(problems) + f". Create a user with readWrite on {s.plan_cache_database} only.")
    else:
        print(f"cache user restricted to {s.plan_cache_database}: OK")


def cmd_verify(_args) -> None:
    client, coll, s = _collection()
    print(f"server {client.server_info().get('version')} · {s.plan_cache_database}.{s.plan_cache_collection}")
    _check_privileges(client, s)
    print("collections:", client[s.plan_cache_database].list_collection_names())
    indexes = list(coll.list_search_indexes())
    found = [i for i in indexes if i.get("name") == s.plan_cache_vector_index]
    if not found:
        print(f"MISSING vector index {s.plan_cache_vector_index!r} — run create-index with an admin credential")
    for i in found:
        print(f"index {i['name']}: status={i.get('status')} queryable={i.get('queryable')}")
        print(json.dumps(i.get("latestDefinition"), indent=1))
    ttl = [i for i in coll.list_indexes() if "expireAfterSeconds" in i]
    print("TTL index:", [(i["name"], dict(i["key"]), i["expireAfterSeconds"]) for i in ttl] or "MISSING (created by the app at startup)")
    print("templates:", {"active": coll.count_documents({"revoked": False}), "revoked": coll.count_documents({"revoked": True}),
                         "without revoked flag (never used)": coll.count_documents({"revoked": {"$exists": False}})})


def cmd_create_index(args) -> None:
    uri = os.environ.get(args.admin_uri_env)
    if not uri:
        sys.exit(f"set the admin connection string in ${args.admin_uri_env} (it is never read from the command line)")
    client, coll, s = _collection(uri)
    if s.plan_cache_collection not in client[s.plan_cache_database].list_collection_names():
        client[s.plan_cache_database].create_collection(s.plan_cache_collection)
    coll.create_index("expires_at", expireAfterSeconds=0, name="expires_at_ttl")  # the app also ensures it at startup
    name = coll.create_search_index(
        SearchIndexModel(definition=index_definition(), name=s.plan_cache_vector_index, type="vectorSearch")
    )
    print(f"requested vector index {name!r} on {s.plan_cache_database}.{s.plan_cache_collection}; "
          "Atlas builds it in the background — check with `verify` until queryable=True")  # fmt: skip


def cmd_update_index(args) -> None:
    uri = os.environ.get(args.admin_uri_env)
    if not uri:
        sys.exit(f"set the admin connection string in ${args.admin_uri_env} (it is never read from the command line)")
    _, coll, s = _collection(uri)
    coll.update_search_index(s.plan_cache_vector_index, index_definition())
    print(f"requested update of {s.plan_cache_vector_index!r}; Atlas rebuilds it — check with `verify` until queryable=True")


def _brief(d: dict) -> str:
    state = "revoked" if d.get("revoked") else "active"
    return (f"{d['_id']}  {state:<8} {d.get('action', ''):<10} {d.get('source', ''):<12} "
            f"seen={d.get('seen_count', 0):<4} used={d.get('usage_count', 0):<4} {d.get('question_text', '')}")  # fmt: skip


def cmd_list(args) -> None:
    _, coll, _ = _collection()
    query = {"revoked": True} if args.revoked else {"revoked": False}
    for d in coll.find(query, {"embedding": 0}).sort("last_seen_at", -1).limit(args.limit):
        print(_brief(d))


def cmd_show(args) -> None:
    _, coll, _ = _collection()
    d = coll.find_one({"_id": args.template_id}, {"embedding": 0})
    if not d:
        sys.exit("no such template")
    print(json.dumps(d, indent=1, default=str))


def cmd_revoke(args) -> None:
    _, coll, _ = _collection()
    now = datetime.now(UTC)
    r = coll.update_one(
        {"_id": args.template_id},
        {"$set": {"revoked": True, "revoked_by": args.by, "revoked_at": now, "revoke_reason": args.reason}},
    )
    print("revoked (takes effect on the next query)" if r.matched_count else "unknown id")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    c = sub.add_parser("create-index")
    c.add_argument("--admin-uri-env", required=True, help="name of the environment variable holding an admin URI")
    c.set_defaults(fn=cmd_create_index)
    u = sub.add_parser("update-index")
    u.add_argument("--admin-uri-env", required=True, help="name of the environment variable holding an admin URI")
    u.set_defaults(fn=cmd_update_index)
    lst = sub.add_parser("list")
    lst.add_argument("--revoked", action="store_true", help="list revoked templates instead of active ones")
    lst.add_argument("--limit", type=int, default=50)
    lst.set_defaults(fn=cmd_list)
    for name, fn in (("show", cmd_show), ("revoke", cmd_revoke)):
        sp = sub.add_parser(name)
        sp.add_argument("template_id")
        if name == "revoke":
            sp.add_argument("--by", required=True)
            sp.add_argument("--reason", required=True)
        sp.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
