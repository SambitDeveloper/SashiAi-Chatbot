import { ResidentRecord, ResidentAttribute, AttributeSpotlight } from '../types/resident';
import { STATUS_LABELS, CARE_TYPE_LABELS } from '../data/residentDatabase';

export interface AttributeDefinition {
  field: ResidentAttribute;
  label: string;
  synonyms: string[];
  questionPatterns: RegExp[];
}

/**
 * Healthcare Staff Semantic Attribute Ontology
 * Maps diverse natural language questions into canonical resident data fields
 */
export const RESIDENT_ATTRIBUTE_ONTOLOGY: AttributeDefinition[] = [
  {
    field: 'primaryCaregiver',
    label: 'Primary Caregiver',
    synonyms: [
      'caretaker', 'care taker', 'caregiver', 'care giver', 'nurse', 'aide', 'care aide',
      'primary caretaker', 'primary care taker', 'primary caregiver', 'primary care giver',
      'registered nurse', 'lpn', 'rn', 'who looks after', 'who takes care of', 'whose care',
      'assigned nurse', 'assigned caretaker', 'assigned care taker', 'attending nurse',
      'nurse in charge', 'case worker', 'care lead'
    ],
    questionPatterns: [
      /who (is|are) (the )?(primary )?(caretaker|care\s*taker|caregiver|care\s*giver|nurse|aide|doctor)/i,
      /who (takes care of|looks after|cares for|manages)/i,
      /whose care (is|was)/i,
      /(primary|assigned) (caretaker|care\s*taker|caregiver|care\s*giver|nurse)/i,
      /\bcare\s*taker\b/i
    ]
  },
  {
    field: 'roomNumber',
    label: 'Room & Location',
    synonyms: [
      'room', 'room number', 'bed', 'unit', 'where is she staying',
      'where is he staying', 'where is she located', 'where is he located',
      'where does she live', 'where does he live', 'which room', 'what room'
    ],
    questionPatterns: [
      /which (room|bed|unit|wing|floor)/i,
      /what (room|bed|unit|number)/i,
      /where (is|does) .* (staying|located|living|sleep|reside)/i,
      /room (number|assignment)/i
    ]
  },
  {
    field: 'wing',
    label: 'Facility Wing',
    synonyms: ['wing', 'building wing', 'facility wing', 'garden terrace', 'north wing', 'south wing', 'east wing'],
    questionPatterns: [
      /which wing/i,
      /what wing/i,
      /which facility area/i
    ]
  },
  {
    field: 'admissionDate',
    label: 'Admission Date',
    synonyms: [
      'admitted', 'admission', 'admission date', 'date of admission',
      'arrival date', 'when did she arrive', 'when did he arrive',
      'when did she move in', 'when did he move in', 'move-in date',
      'intake date', 'registration date'
    ],
    questionPatterns: [
      /when was .* (admitted|registered|brought in|enrolled)/i,
      /when did .* (move in|arrive|enter|join)/i,
      /(admission|admit|move-in|arrival) date/i
    ]
  },
  {
    field: 'emergencyContact',
    label: 'Emergency Contact',
    synonyms: [
      'emergency contact', 'next of kin', 'family contact', 'who to call',
      'phone number', 'contact person', 'daughter', 'son', 'spouse', 'guardian'
    ],
    questionPatterns: [
      /who is (the )?(emergency contact|next of kin|family contact)/i,
      /who (to call|can we reach|can i call)/i,
      /(emergency|family) (contact|phone|number)/i,
      /phone number for/i
    ]
  },
  {
    field: 'careType',
    label: 'Care Category / Level',
    synonyms: [
      'care type', 'category', 'care level', 'care category',
      'level of care', 'what kind of care', 'service tier'
    ],
    questionPatterns: [
      /(what|which) (category|care type|care level|level of care)/i,
      /what kind of care/i,
      /is .* in (memory care|assisted living|skilled nursing|independent|respite)/i
    ]
  },
  {
    field: 'status',
    label: 'Resident Status',
    synonyms: [
      'status', 'current status', 'active', 'inactive', 'pending',
      'discharged', 'is she active', 'is he active', 'is she still here'
    ],
    questionPatterns: [
      /(what is|tell me) .* (status|current status)/i,
      /is .* (active|inactive|pending|discharged|still living here|still here)/i
    ]
  },
  {
    field: 'age',
    label: 'Age',
    synonyms: ['age', 'how old', 'years old', 'birth date', 'dob'],
    questionPatterns: [
      /how old is/i,
      /what is .* age/i
    ]
  }
];

export interface DetectedTargetAttribute {
  attribute: ResidentAttribute;
  confidence: number;
  label: string;
  matchIndex: number;
}

/**
 * Detects ALL target attributes present in compound user queries
 * (e.g. "Donald Draper belongs to which care level and care taker ?")
 */
export function detectTargetAttributes(query: string): DetectedTargetAttribute[] {
  const normalized = query.toLowerCase().trim();
  const results: DetectedTargetAttribute[] = [];
  const seen = new Set<ResidentAttribute>();

  for (const def of RESIDENT_ATTRIBUTE_ONTOLOGY) {
    let bestIndex = -1;
    let confidence = 0;

    // 1. Check regex question patterns (high confidence)
    for (const pattern of def.questionPatterns) {
      const match = normalized.match(pattern);
      if (match && match.index !== undefined) {
        if (bestIndex === -1 || match.index < bestIndex) {
          bestIndex = match.index;
          confidence = 0.98;
        }
      }
    }

    // 2. Check synonym keywords in query
    for (const syn of def.synonyms) {
      const regex = new RegExp(`\\b${syn.replace(/\s+/g, '\\s+')}\\b`, 'i');
      const match = normalized.match(regex);
      if (match && match.index !== undefined) {
        if (bestIndex === -1 || match.index < bestIndex) {
          bestIndex = match.index;
          confidence = Math.max(confidence, 0.92);
        }
      }
    }

    if (bestIndex !== -1 && !seen.has(def.field)) {
      seen.add(def.field);
      results.push({
        attribute: def.field,
        confidence,
        label: def.label,
        matchIndex: bestIndex
      });
    }
  }

  // Sort by order of appearance in query
  return results.sort((a, b) => a.matchIndex - b.matchIndex);
}

/**
 * Detects if user query is specifically seeking a single attribute of a resident
 */
export function detectTargetAttribute(query: string): DetectedTargetAttribute | null {
  const all = detectTargetAttributes(query);
  return all.length > 0 ? all[0] : null;
}

/**
 * Formats a clean, direct spotlight card for the requested target attribute
 */
export function formatAttributeAnswer(resident: ResidentRecord, attribute: ResidentAttribute): AttributeSpotlight {
  switch (attribute) {
    case 'primaryCaregiver':
      return {
        attribute,
        label: 'Primary Caregiver',
        value: resident.primaryCaregiver,
        supportingDetail: `Assigned primary clinical staff member for ${resident.firstName} ${resident.lastName} in ${resident.roomNumber} (${resident.wing}).`
      };

    case 'roomNumber':
      return {
        attribute,
        label: 'Room & Location',
        value: `${resident.roomNumber} · ${resident.wing}`,
        supportingDetail: `${resident.firstName} is residing in Room ${resident.roomNumber} on the ${resident.wing} (${CARE_TYPE_LABELS[resident.careType]}).`
      };

    case 'wing':
      return {
        attribute,
        label: 'Facility Wing',
        value: resident.wing,
        supportingDetail: `Room ${resident.roomNumber} is located in the ${resident.wing}.`
      };

    case 'admissionDate':
      return {
        attribute,
        label: 'Admission Date',
        value: resident.admissionDate,
        supportingDetail: `${resident.firstName} ${resident.lastName} was formally admitted on ${resident.admissionDate}.`
      };

    case 'emergencyContact':
      return {
        attribute,
        label: 'Emergency Contact',
        value: `${resident.emergencyContact.name} (${resident.emergencyContact.relationship})`,
        supportingDetail: `Contact Phone: ${resident.emergencyContact.phone}`
      };

    case 'careType':
      return {
        attribute,
        label: 'Care Category',
        value: CARE_TYPE_LABELS[resident.careType] || resident.careType,
        supportingDetail: `${resident.firstName} is enrolled in the ${CARE_TYPE_LABELS[resident.careType]} program.`
      };

    case 'status':
      return {
        attribute,
        label: 'Resident Status',
        value: STATUS_LABELS[resident.status] || resident.status,
        supportingDetail: `Current record census status in Care App database is "${STATUS_LABELS[resident.status]}".`
      };

    case 'age':
      return {
        attribute,
        label: 'Resident Age',
        value: `${resident.age} years old`,
        supportingDetail: `${resident.firstName} ${resident.lastName} (${resident.gender}).`
      };

    default:
      return {
        attribute: 'fullProfile',
        label: 'Resident Overview',
        value: `${resident.firstName} ${resident.lastName}`,
        supportingDetail: `${STATUS_LABELS[resident.status]} · ${CARE_TYPE_LABELS[resident.careType]} in Room ${resident.roomNumber}`
      };
  }
}
