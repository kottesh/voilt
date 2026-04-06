import React, { useState, useEffect, useMemo } from 'react';
import { Search, X, RefreshCcw, Image as ImageIcon } from 'lucide-react';

// --- Types ---
type StatusType = 'pending' | 'invested' | 'skipped';

interface VehicleRecord {
  id: string;
  number_plate: string;
  confidence_level: number;
  image_url: string;
  date: string;
  status: StatusType;
}

// --- Sub-Component: Status Badge ---
const StatusBadge = ({ status }: { status: StatusType }) => {
  const styles = {
    pending: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    invested: 'bg-blue-100 text-blue-800 border-blue-200',
    skipped: 'bg-slate-100 text-slate-600 border-slate-200'
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${styles[status]} capitalize`}>
      {status}
    </span>
  );
};

// --- Sub-Component: Image Modal ---
const ImageModal = ({ isOpen, imageUrl, plate, onClose }: { isOpen: boolean; imageUrl: string; plate: string; onClose: () => void }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="relative bg-white rounded-lg shadow-2xl overflow-hidden max-w-2xl w-full animate-in fade-in zoom-in duration-200" onClick={e => e.stopPropagation()}>
        <div className="p-3 border-b flex justify-between items-center bg-slate-50">
          <h3 className="font-semibold text-slate-700">Image: {plate}</h3>
          <button onClick={onClose} className="p-1 hover:bg-slate-200 rounded-full transition-colors">
            <X className="w-5 h-5 text-slate-500" />
          </button>
        </div>
        <div className="p-2 flex justify-center bg-slate-100">
          <img src={imageUrl} alt={plate} className="max-h-[60vh] object-contain rounded shadow-sm" />
        </div>
        <div className="p-3 border-t bg-slate-50 text-right">
          <button 
            onClick={onClose}
            className="px-4 py-2 bg-white border border-slate-300 rounded text-sm font-medium hover:bg-slate-50 transition-colors text-slate-700"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export const VehicleTable = () => {
  // -- State --
  const [allData, setAllData] = useState<VehicleRecord[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterDate, setFilterDate] = useState('');
  const [filterConfidence, setFilterConfidence] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  
  // Modal State
  const [modalImage, setModalImage] = useState<{ url: string; plate: string } | null>(null);

  const itemsPerPage = 7;

  // -- Initialization: Generate Dummy Data --
  useEffect(() => {
    const generateDummyData = (): VehicleRecord[] => {
      const statuses: StatusType[] = ['pending', 'invested', 'skipped'];
      const data: VehicleRecord[] = [];
      
      for (let i = 1; i <= 55; i++) {
        const randomDate = new Date(2023, Math.floor(Math.random() * 12), Math.floor(Math.random() * 28) + 1);
        const dateStr = randomDate.toISOString().split('T')[0];
        
        data.push({
          id: `REC-${1000 + i}`,
          number_plate: `${String.fromCharCode(65 + Math.floor(Math.random() * 26))}${Math.floor(Math.random() * 9000 + 1000)} ${String.fromCharCode(65 + Math.floor(Math.random() * 26))}${String.fromCharCode(65 + Math.floor(Math.random() * 26))}`,
          confidence_level: Math.floor(Math.random() * 40 + 60), // 60-100%
          image_url: `https://picsum.photos/seed/${i + 100}/400/300`,
          date: dateStr,
          status: statuses[Math.floor(Math.random() * statuses.length)]
        });
      }
      return data.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
    };

    // Simulating an API fetch
    setAllData(generateDummyData());
  }, []);

  // -- Filtering Logic --
  const filteredData = useMemo(() => {
    return allData.filter(item => {
      const matchesSearch = item.number_plate.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesDate = filterDate ? item.date === filterDate : true;
      const matchesConfidence = filterConfidence ? item.confidence_level >= parseInt(filterConfidence) : true;
      return matchesSearch && matchesDate && matchesConfidence;
    });
  }, [allData, searchTerm, filterDate, filterConfidence]);

  // -- Pagination Logic --
  const totalPages = Math.ceil(filteredData.length / itemsPerPage);
  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return filteredData.slice(startIndex, startIndex + itemsPerPage);
  }, [filteredData, currentPage]);

  // Reset page on filter change
  useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm, filterDate, filterConfidence]);

  // -- Handlers --
  const handleStatusChange = (id: string, newStatus: StatusType) => {
    setAllData(prev => prev.map(item => 
      item.id === id ? { ...item, status: newStatus } : item
    ));
  };

  const openModal = (url: string, plate: string) => setModalImage({ url, plate });
  const closeModal = () => setModalImage(null);

  return (
    <div className="max-w-7xl mx-auto p-4 md:p-8">
      
      {/* Header */}
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 tracking-tight">Vehicle Recognition Log</h1>
        <p className="text-slate-500 mt-1">Manage and investigate automated license plate records.</p>
      </header>

      {/* Controls / Filters */}
      <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
          
          {/* Search by Plate */}
          <div className="md:col-span-5">
            <label className="block text-xs font-semibold text-slate-500 uppercase mb-1">Number Plate</label>
            <div className="relative">
              <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
              <input 
                type="text" 
                placeholder="Search plate..." 
                className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>

          {/* Filter by Date */}
          <div className="md:col-span-3">
            <label className="block text-xs font-semibold text-slate-500 uppercase mb-1">Date</label>
            <input 
              type="date" 
              className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow"
              value={filterDate}
              onChange={(e) => setFilterDate(e.target.value)}
            />
          </div>

          {/* Filter by Confidence */}
          <div className="md:col-span-2">
            <label className="block text-xs font-semibold text-slate-500 uppercase mb-1">Min Confidence %</label>
            <input 
              type="number" 
              min="0" 
              max="100" 
              placeholder="> 0" 
              className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow"
              value={filterConfidence}
              onChange={(e) => setFilterConfidence(e.target.value)}
            />
          </div>

          {/* Reset */}
          <div className="md:col-span-2">
            <button 
              onClick={() => { setSearchTerm(''); setFilterDate(''); setFilterConfidence(''); }}
              className="w-full py-2 px-4 border border-slate-300 text-slate-600 rounded-lg text-sm font-medium hover:bg-slate-50 hover:text-slate-800 transition-colors flex items-center justify-center gap-2"
            >
              <RefreshCcw className="w-4 h-4" /> Reset
            </button>
          </div>
        </div>
      </div>

      {/* Table Card */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-xs uppercase text-slate-500 font-semibold tracking-wider">
                <th className="p-4">ID</th>
                <th className="p-4">Number Plate</th>
                <th className="p-4">Confidence</th>
                <th className="p-4">Status</th>
                <th className="p-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {paginatedData.length > 0 ? (
                paginatedData.map((record) => (
                  <tr key={record.id} className="hover:bg-blue-50/50 transition-colors group">
                    <td className="p-4">
                      <span className="text-slate-400 text-xs font-mono">{record.id}</span>
                      <div className="text-[10px] text-slate-400">{record.date}</div>
                    </td>
                    <td className="p-4">
                      <span className="font-mono font-medium text-slate-800 bg-slate-100 px-2 py-1 rounded">
                        {record.number_plate}
                      </span>
                    </td>
                    <td className="p-4">
                      <div className="flex items-center gap-3">
                        <div className="flex-1 w-24 bg-slate-200 rounded-full h-2 overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${record.confidence_level > 85 ? 'bg-emerald-500' : record.confidence_level > 70 ? 'bg-yellow-500' : 'bg-red-500'}`} 
                            style={{ width: `${record.confidence_level}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-slate-600 w-8 text-right">{record.confidence_level}%</span>
                      </div>
                    </td>
                    <td className="p-4">
                      <select
                        value={record.status}
                        onChange={(e) => handleStatusChange(record.id, e.target.value as StatusType)}
                        className={`text-xs font-bold uppercase border rounded px-2 py-1 cursor-pointer focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-blue-500
                          ${record.status === 'pending' ? 'bg-yellow-50 text-yellow-700 border-yellow-200' : 
                            record.status === 'invested' ? 'bg-blue-50 text-blue-700 border-blue-200' : 
                            'bg-slate-50 text-slate-600 border-slate-200'}`}
                      >
                        <option value="pending">Pending</option>
                        <option value="invested">Invested</option>
                        <option value="skipped">Skipped</option>
                      </select>
                    </td>
                    <td className="p-4 text-right">
                      <button 
                        onClick={() => openModal(record.image_url, record.number_plate)}
                        className="text-blue-600 hover:text-blue-800 text-sm font-medium flex items-center justify-end gap-1 ml-auto transition-colors"
                      >
                        <ImageIcon className="w-4 h-4" /> View
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="p-8 text-center text-slate-500">
                    No records found matching your filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="p-4 border-t border-slate-200 flex items-center justify-between bg-slate-50/50">
          <span className="text-sm text-slate-500">
            Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(currentPage * itemsPerPage, filteredData.length)} of {filteredData.length} results
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1 text-sm border border-slate-300 rounded bg-white text-slate-600 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-100 transition-colors"
            >
              Previous
            </button>
            <div className="flex items-center px-2">
              <span className="text-sm font-medium text-slate-700">Page {currentPage} of {totalPages || 1}</span>
            </div>
            <button
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages || totalPages === 0}
              className="px-3 py-1 text-sm border border-slate-300 rounded bg-white text-slate-600 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-100 transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {/* Modal Render */}
      <ImageModal 
        isOpen={!!modalImage} 
        imageUrl={modalImage?.url || ''} 
        plate={modalImage?.plate || ''} 
        onClose={closeModal} 
      />
    </div>
  );
};