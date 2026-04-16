import { useState, useEffect } from 'react';
import { 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  TrendingUp, 
  Camera,
  Search,
  Filter,
  Download,
  RefreshCw,
  Eye,
  X,
  Calendar,
  MapPin
} from 'lucide-react';
import { violationsService, type Violation } from '../services/violationsService';

// Stats Card Component
const StatCard = ({ 
  title, 
  value, 
  icon: Icon, 
  trend, 
  color = 'blue' 
}: { 
  title: string; 
  value: string | number; 
  icon: any; 
  trend?: string; 
  color?: string;
}) => {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600 border-blue-200',
    red: 'bg-red-50 text-red-600 border-red-200',
    green: 'bg-green-50 text-green-600 border-green-200',
    yellow: 'bg-yellow-50 text-yellow-600 border-yellow-200',
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-500 mb-1">{title}</p>
          <p className="text-3xl font-bold text-slate-900">{value}</p>
          {trend && (
            <p className="text-xs text-slate-500 mt-2 flex items-center gap-1">
              <TrendingUp className="w-3 h-3" /> {trend}
            </p>
          )}
        </div>
        <div className={`p-3 rounded-lg ${colorClasses[color as keyof typeof colorClasses]}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  );
};

// Violation Detail Modal
const ViolationModal = ({ 
  violation, 
  onClose 
}: { 
  violation: Violation | null; 
  onClose: () => void;
}) => {
  if (!violation) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="relative bg-white rounded-xl shadow-2xl overflow-hidden max-w-4xl w-full animate-in fade-in zoom-in duration-200" onClick={e => e.stopPropagation()}>
        <div className="p-4 border-b flex justify-between items-center bg-gradient-to-r from-slate-50 to-slate-100">
          <div>
            <h3 className="font-bold text-xl text-slate-800">Violation Details</h3>
            <p className="text-sm text-slate-500">ID: {violation.id}</p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-slate-200 rounded-full transition-colors">
            <X className="w-5 h-5 text-slate-500" />
          </button>
        </div>
        
        <div className="grid md:grid-cols-2 gap-6 p-6">
          {/* Image Section */}
          <div className="space-y-4">
            {violation.evidence_image_url ? (
              <div className="relative rounded-lg overflow-hidden bg-slate-100 border-2 border-slate-200">
                <img 
                  src={violation.evidence_image_url} 
                  alt="Evidence" 
                  className="w-full h-auto object-contain max-h-96"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iI2YxZjVmOSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzk0YTNiOCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlIG5vdCBhdmFpbGFibGU8L3RleHQ+PC9zdmc+';
                  }}
                />
                <div className="absolute top-2 right-2 bg-black/70 text-white px-3 py-1 rounded-full text-xs font-medium">
                  Evidence Photo
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 bg-slate-100 rounded-lg border-2 border-dashed border-slate-300">
                <div className="text-center">
                  <Camera className="w-12 h-12 text-slate-400 mx-auto mb-2" />
                  <p className="text-slate-500">No image available</p>
                </div>
              </div>
            )}
          </div>

          {/* Details Section */}
          <div className="space-y-4">
            <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
              <h4 className="text-sm font-semibold text-slate-500 uppercase mb-3">Violation Information</h4>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center py-2 border-b border-slate-200">
                  <span className="text-sm text-slate-600">Number Plate</span>
                  <span className="font-bold text-lg font-mono bg-white px-3 py-1 rounded border border-slate-300">
                    {violation.number_plate || 'Not detected'}
                  </span>
                </div>

                <div className="flex justify-between items-center py-2 border-b border-slate-200">
                  <span className="text-sm text-slate-600">Confidence</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 bg-slate-200 rounded-full h-2">
                      <div 
                        className={`h-full rounded-full ${
                          violation.confidence_level >= 0.9 ? 'bg-green-500' : 
                          violation.confidence_level >= 0.7 ? 'bg-yellow-500' : 
                          'bg-red-500'
                        }`}
                        style={{ width: `${violation.confidence_level * 100}%` }}
                      />
                    </div>
                    <span className="font-bold text-sm">
                      {(violation.confidence_level * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className="flex justify-between items-center py-2 border-b border-slate-200">
                  <span className="text-sm text-slate-600">Status</span>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase ${
                    violation.status === 'confirmed' ? 'bg-green-100 text-green-700' :
                    violation.status === 'pending' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-slate-100 text-slate-600'
                  }`}>
                    {violation.status}
                  </span>
                </div>

                <div className="flex justify-between items-center py-2 border-b border-slate-200">
                  <span className="text-sm text-slate-600 flex items-center gap-1">
                    <Calendar className="w-4 h-4" /> Captured At
                  </span>
                  <span className="font-medium text-sm">
                    {new Date(violation.captured_at).toLocaleString()}
                  </span>
                </div>

                <div className="flex justify-between items-center py-2 border-b border-slate-200">
                  <span className="text-sm text-slate-600">Created At</span>
                  <span className="font-medium text-sm">
                    {new Date(violation.created_at).toLocaleString()}
                  </span>
                </div>

                {violation.camera_id && (
                  <div className="flex justify-between items-center py-2">
                    <span className="text-sm text-slate-600 flex items-center gap-1">
                      <MapPin className="w-4 h-4" /> Camera ID
                    </span>
                    <span className="font-mono text-sm bg-white px-2 py-1 rounded border border-slate-300">
                      {violation.camera_id}
                    </span>
                  </div>
                )}
              </div>
            </div>

            <div className="flex gap-2">
              <button className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors">
                Generate Ticket
              </button>
              <button className="flex-1 bg-slate-200 text-slate-700 px-4 py-2 rounded-lg font-medium hover:bg-slate-300 transition-colors">
                Mark as False
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Dashboard Component
export const Dashboard = () => {
  const [violations, setViolations] = useState<Violation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [selectedViolation, setSelectedViolation] = useState<Violation | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const fetchViolations = async () => {
    setLoading(true);
    setError(null);
    try {
      const filterStatus = statusFilter === 'all' ? null : statusFilter;
      const response = await violationsService.getViolations(filterStatus, 500, 0);
      setViolations(response.violations);
    } catch (err) {
      console.error('Failed to fetch violations:', err);
      setError('Failed to load violations. Please check if the server is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchViolations();
  }, [statusFilter]);

  // Calculate stats
  const stats = {
    total: violations.length,
    confirmed: violations.filter(v => v.status === 'confirmed').length,
    pending: violations.filter(v => v.status === 'pending').length,
    withPlate: violations.filter(v => v.number_plate).length,
  };

  // Filter violations
  const filteredViolations = violations.filter(v => {
    const matchesSearch = !searchTerm || 
      (v.number_plate?.toLowerCase().includes(searchTerm.toLowerCase())) ||
      v.id.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesSearch;
  });

  // Paginate
  const totalPages = Math.ceil(filteredViolations.length / itemsPerPage);
  const paginatedViolations = filteredViolations.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
                <AlertTriangle className="w-8 h-8 text-red-600" />
                Traffic Violation Dashboard
              </h1>
              <p className="text-slate-500 mt-1">Real-time monitoring and management</p>
            </div>
            <button 
              onClick={fetchViolations}
              disabled={loading}
              className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 mb-8">
          <StatCard 
            title="Total Violations" 
            value={stats.total}
            icon={AlertTriangle}
            trend="Last 30 days"
            color="blue"
          />
          <StatCard 
            title="Confirmed" 
            value={stats.confirmed}
            icon={CheckCircle}
            trend={`${((stats.confirmed/stats.total)*100 || 0).toFixed(1)}% of total`}
            color="green"
          />
          <StatCard 
            title="Pending Review" 
            value={stats.pending}
            icon={Clock}
            trend="Needs attention"
            color="yellow"
          />
          <StatCard 
            title="Plates Detected" 
            value={stats.withPlate}
            icon={Camera}
            trend={`${((stats.withPlate/stats.total)*100 || 0).toFixed(1)}% success rate`}
            color="blue"
          />
        </div>

        {/* Filters */}
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-4 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
            <div className="md:col-span-6">
              <div className="relative">
                <Search className="absolute left-3 top-3 w-5 h-5 text-slate-400" />
                <input
                  type="text"
                  placeholder="Search by plate number or ID..."
                  className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
            </div>
            <div className="md:col-span-3">
              <div className="relative">
                <Filter className="absolute left-3 top-3 w-5 h-5 text-slate-400" />
                <select
                  className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 appearance-none"
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                >
                  <option value="all">All Status</option>
                  <option value="confirmed">Confirmed</option>
                  <option value="pending">Pending</option>
                  <option value="mailed">Mailed</option>
                </select>
              </div>
            </div>
            <div className="md:col-span-3">
              <button className="w-full py-2.5 px-4 bg-slate-700 text-white rounded-lg font-medium hover:bg-slate-800 transition-colors flex items-center justify-center gap-2">
                <Download className="w-4 h-4" />
                Export Data
              </button>
            </div>
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-4 mb-6 rounded">
            <p className="font-medium">{error}</p>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-slate-500">Loading violations...</p>
          </div>
        )}

        {/* Violations Table */}
        {!loading && !error && (
          <div className="bg-white rounded-lg shadow-sm border border-slate-200 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Date & Time
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Number Plate
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Confidence
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Evidence
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {paginatedViolations.length > 0 ? (
                    paginatedViolations.map((violation) => (
                      <tr key={violation.id} className="hover:bg-blue-50/30 transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-slate-900">
                            {new Date(violation.captured_at).toLocaleDateString()}
                          </div>
                          <div className="text-xs text-slate-500">
                            {new Date(violation.captured_at).toLocaleTimeString()}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {violation.number_plate ? (
                            <span className="inline-flex items-center px-3 py-1 rounded-md text-sm font-bold font-mono bg-slate-800 text-white">
                              {violation.number_plate}
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-3 py-1 rounded-md text-xs font-medium bg-slate-100 text-slate-500">
                              Not detected
                            </span>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center gap-2">
                            <div className="w-24 bg-slate-200 rounded-full h-2">
                              <div
                                className={`h-full rounded-full ${
                                  violation.confidence_level >= 0.9 ? 'bg-green-500' :
                                  violation.confidence_level >= 0.7 ? 'bg-yellow-500' :
                                  'bg-red-500'
                                }`}
                                style={{ width: `${violation.confidence_level * 100}%` }}
                              />
                            </div>
                            <span className="text-sm font-medium text-slate-700">
                              {(violation.confidence_level * 100).toFixed(0)}%
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                            violation.status === 'confirmed' ? 'bg-green-100 text-green-800' :
                            violation.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-slate-100 text-slate-800'
                          }`}>
                            {violation.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {violation.evidence_image_url ? (
                            <span className="inline-flex items-center text-green-600 text-sm">
                              <CheckCircle className="w-4 h-4 mr-1" />
                              Available
                            </span>
                          ) : (
                            <span className="inline-flex items-center text-slate-400 text-sm">
                              <X className="w-4 h-4 mr-1" />
                              Missing
                            </span>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                          <button
                            onClick={() => setSelectedViolation(violation)}
                            className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-900 font-medium"
                          >
                            <Eye className="w-4 h-4" />
                            View Details
                          </button>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="px-6 py-12 text-center text-slate-500">
                        No violations found matching your criteria.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="bg-slate-50 px-6 py-4 border-t border-slate-200 flex items-center justify-between">
                <div className="text-sm text-slate-600">
                  Showing <span className="font-medium">{(currentPage - 1) * itemsPerPage + 1}</span> to{' '}
                  <span className="font-medium">{Math.min(currentPage * itemsPerPage, filteredViolations.length)}</span> of{' '}
                  <span className="font-medium">{filteredViolations.length}</span> results
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="px-4 py-2 border border-slate-300 rounded-lg text-sm font-medium text-slate-700 bg-white hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    Previous
                  </button>
                  <div className="flex items-center px-4">
                    <span className="text-sm font-medium text-slate-700">
                      Page {currentPage} of {totalPages}
                    </span>
                  </div>
                  <button
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="px-4 py-2 border border-slate-300 rounded-lg text-sm font-medium text-slate-700 bg-white hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Violation Detail Modal */}
      <ViolationModal 
        violation={selectedViolation}
        onClose={() => setSelectedViolation(null)}
      />
    </div>
  );
};
