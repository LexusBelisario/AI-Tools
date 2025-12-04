import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { FiLogOut, FiUser, FiChevronDown, FiBell } from "react-icons/fi";
import { API_URL } from "../../config";

const AdminHeader = () => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);
  const location = useLocation();
  const navigate = useNavigate();

  const getPageTitle = () => {
    const path = location.pathname;
    if (path.includes("dashboard")) return "Dashboard";
    if (path.includes("users")) return "User Management";
    if (path.includes("data-upload")) return "CSV/SHP File Upload";
    return "Dashboard";
  };

  const fetchPendingUsers = async () => {
    try {
      const res = await fetch(`${API_URL}/api/admin/users/pending`);
      if (res.ok) {
        const data = await res.json();
        setPendingCount(data.total_pending || 0);
      }
    } catch (err) {
      console.error("Error fetching pending users:", err);
    }
  };

  useEffect(() => {
    fetchPendingUsers();
    const interval = setInterval(fetchPendingUsers, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("accessToken");
    navigate("/login");
  };

  return (
    <header className="bg-white border-b border-[#E0E0E0]">
      <div className="px-6 py-4 flex justify-between items-center">
        <h1 className="text-2xl font-extrabold text-[#222222]">
          {getPageTitle()}
        </h1>

        <div className="flex items-center gap-6">
          <div className="relative">
            <FiBell className="text-gray-600 text-xl" />
            {pendingCount > 0 && (
              <span className="absolute -top-2 -right-2 bg-red-600 text-white text-xs rounded-full px-2 py-0.5">
                {pendingCount}
              </span>
            )}
          </div>

          <div className="relative">
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center space-x-3 hover:bg-gray-50 rounded-lg px-3 py-2 transition-colors"
            >
              <div className="w-10 h-10 bg-[#00519C] rounded-full flex items-center justify-center">
                <FiUser className="text-white text-lg" />
              </div>
              <div className="text-left">
                <p className="text-sm font-bold text-[#222222]">
                  Juan Dela Cruz
                </p>
                <p className="text-xs font-bold text-[#777777]">Admin</p>
              </div>
              <FiChevronDown
                className={`text-gray-400 transition-transform ${
                  dropdownOpen ? "rotate-180" : ""
                }`}
              />
            </button>

            {dropdownOpen && (
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg border border-[#E0E0E0] shadow-sm py-1">
                <button
                  onClick={handleLogout}
                  className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 flex items-center space-x-2 transition-colors"
                >
                  <FiLogOut className="text-[#E90000]" />
                  <span>Logout</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default AdminHeader;
