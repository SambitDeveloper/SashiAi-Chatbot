import React, { useEffect, useState } from 'react';
import { STATUS_LABELS, CARE_TYPE_LABELS } from '../data/labels';
import { ResidentRecord } from '../types/resident';
import { api, ApiError } from '../api/client';
import { X, Search, Database, Filter } from 'lucide-react';

interface DatabaseModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DatabaseModal: React.FC<DatabaseModalProps> = ({ isOpen, onClose }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [records, setRecords] = useState<ResidentRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    setError(null);
    api
      .residents()
      .then(r => setRecords(r.residents))
      .catch((err: unknown) => setError(err instanceof ApiError ? err.message : 'Could not load resident records.'))
      .finally(() => setLoading(false));
  }, [isOpen]);

  if (!isOpen) return null;

  const filtered = records.filter(r => {
    const matchesSearch = 
      r.firstName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.lastName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.residentId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.roomNumber.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesStatus = statusFilter === 'all' || r.status === statusFilter;
    const matchesType = typeFilter === 'all' || r.careType === typeFilter;

    return matchesSearch && matchesStatus && matchesType;
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs">
      <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-2xl w-full max-w-5xl h-[85vh] flex flex-col text-slate-200">
        
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-950/70">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-slate-100">Resident Records</h2>
              <p className="text-xs text-slate-400">
                {loading ? 'Loading…' : error ? 'Unavailable' : `${records.length} total records`}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-md hover:bg-slate-800 transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Filter and Search Bar */}
        <div className="p-4 border-b border-slate-800 bg-slate-950/40 flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search by name, resident ID, or room number..."
              className="w-full pl-9 pr-3 py-1.5 bg-slate-900 border border-slate-700/80 rounded-md text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-teal-500"
            />
          </div>

          <div className="flex items-center gap-2 text-xs">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-900 border border-slate-700/80 rounded-md py-1.5 px-2.5 text-slate-300 focus:outline-none focus:border-teal-500 cursor-pointer"
            >
              <option value="all">All Statuses</option>
              <option value="active">Active</option>
              <option value="pending">Pending Intake</option>
              <option value="admitted">Recently Admitted</option>
              <option value="inactive">Inactive</option>
              <option value="discharged">Discharged</option>
            </select>

            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              className="bg-slate-900 border border-slate-700/80 rounded-md py-1.5 px-2.5 text-slate-300 focus:outline-none focus:border-teal-500 cursor-pointer"
            >
              <option value="all">All Care Types</option>
              <option value="assisted_living">Assisted Living</option>
              <option value="memory_care">Memory Care</option>
              <option value="independent_living">Independent Living</option>
              <option value="skilled_nursing">Skilled Nursing</option>
              <option value="respite_care">Respite Care</option>
            </select>
          </div>

          <span className="text-xs font-mono text-slate-400">
            Showing <strong className="text-teal-400">{filtered.length}</strong> of {records.length}
          </span>
        </div>

        {/* Table View */}
        <div className="flex-1 overflow-auto p-4">
          {error && (
            <div className="p-4 rounded-lg bg-amber-950/30 border border-amber-900/40 text-xs text-amber-200">{error}</div>
          )}
          {!error && <table className="w-full text-left text-xs border border-slate-800 rounded-lg">
            <thead className="bg-slate-950 text-slate-400 border-b border-slate-800 sticky top-0">
              <tr>
                <th className="py-2.5 px-3 font-medium">ID</th>
                <th className="py-2.5 px-3 font-medium">Resident Name</th>
                <th className="py-2.5 px-3 font-medium">Room & Wing</th>
                <th className="py-2.5 px-3 font-medium">Status</th>
                <th className="py-2.5 px-3 font-medium">Care Level</th>
                <th className="py-2.5 px-3 font-medium">Admission Date</th>
                <th className="py-2.5 px-3 font-medium">Primary Caregiver</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 bg-slate-900/60">
              {filtered.map((r: ResidentRecord) => (
                <tr key={r.id} className="hover:bg-slate-800/60 transition-colors">
                  <td className="py-2.5 px-3 font-mono text-teal-400 font-semibold">{r.residentId}</td>
                  <td className="py-2.5 px-3 font-medium text-slate-100">
                    {r.firstName} {r.lastName}
                    <span className="text-[11px] text-slate-400 ml-1.5 font-normal">({r.age}y, {r.gender})</span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-300">
                    <span className="font-mono text-slate-200">{r.roomNumber}</span>
                    <span className="text-slate-500 mx-1">·</span>
                    <span className="text-slate-400">{r.wing}</span>
                  </td>
                  <td className="py-2.5 px-3">
                    <span className="capitalize font-medium text-slate-200">
                      {STATUS_LABELS[r.status] || r.status}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-300">
                    {CARE_TYPE_LABELS[r.careType] || r.careType}
                  </td>
                  <td className="py-2.5 px-3 font-mono tabular-nums text-slate-300">
                    {r.admissionDate}
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">
                    {r.primaryCaregiver}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/80 flex items-center justify-between text-xs text-slate-400">
          <span>Emergency contact details are never included in bulk views.</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-md transition-colors cursor-pointer"
          >
            Close Viewer
          </button>
        </div>
      </div>
    </div>
  );
};
