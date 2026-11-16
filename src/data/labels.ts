import { ResidentStatus, ResidentCareType } from '../types/resident';

/** Display labels. Resident records themselves live only on the backend. */
export const STATUS_LABELS: Record<ResidentStatus, string> = {
  active: 'Active',
  away: 'Away',
  inactive: 'Inactive',
  pending: 'Pending Intake',
  admitted: 'Recently Admitted',
  discharged: 'Discharged'
};

export const CARE_TYPE_LABELS: Record<ResidentCareType, string> = {
  assisted_living: 'Assisted Living',
  memory_care: 'Memory Care',
  independent_living: 'Independent Living',
  skilled_nursing: 'Skilled Nursing',
  respite_care: 'Respite Care'
};
