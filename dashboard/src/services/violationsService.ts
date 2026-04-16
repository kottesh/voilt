import ky from 'ky';

// Define the types for our violation data
export interface Violation {
  id: string;
  number_plate: string | null;
  confidence_level: number;
  evidence_image_url: string | null;
  camera_id: string | null;
  captured_at: string;
  status: string;
  created_at: string;
}

export interface ViolationsResponse {
  violations: Violation[];
  total: number;
  limit: number;
  offset: number;
}

// Create a Ky instance with base URL
const api = ky.create({
  prefix: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 10000,
  retry: {
    limit: 2,
    methods: ['get', 'post', 'put', 'delete', 'head', 'options'],
    statusCodes: [408, 413, 429, 500, 502, 503, 504],
    backoffLimit: 0,
  },
});

export const violationsService = {
  // Fetch violations with pagination and filtering
  getViolations: async (
    status: string | null = null,
    limit: number = 50,
    offset: number = 0
  ): Promise<ViolationsResponse> => {
    const searchParams = new URLSearchParams();
    if (status) searchParams.append('status', status);
    searchParams.append('limit', limit.toString());
    searchParams.append('offset', offset.toString());
    
     const response = await api.get(`api/violations?${searchParams.toString()}`).json();
    return response as ViolationsResponse;
  },
  
  // Fetch a single violation by ID
  getViolationById: async (id: string): Promise<Violation> => {
     const response = await api.get(`api/violations/${id}`).json();
    return response as Violation;
  },
  
  // Get total count of violations (optional helper)
  getTotalCount: async (status: string | null = null): Promise<number> => {
    const searchParams = new URLSearchParams();
    if (status) searchParams.append('status', status);
    
    // This would require a separate endpoint for count, but we can derive from getViolations
     const response = await api.get(`api/violations?${searchParams.toString()}&limit=1&offset=0`).json() as ViolationsResponse;
    return response.total;
  }
};

export default violationsService;