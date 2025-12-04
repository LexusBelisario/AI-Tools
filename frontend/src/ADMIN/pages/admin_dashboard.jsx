import React, { useState, useEffect } from "react";
import { FiUsers, FiMapPin, FiActivity, FiFileText, FiClock, FiCheckCircle, FiXCircle } from "react-icons/fi";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { API_URL } from "../../config";
import { useNavigate } from "react-router-dom";

const AdminDashboard = () => {
  const navigate = useNavigate();
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [recentActivities, setRecentActivities] = useState([]);

const fetchStatistics = async () => {
  try {
    const token = localStorage.getItem("access_token");
    console.log("Token:", token); // Debug log
    
    const res = await fetch(`${API_URL}/api/admin/statistics`, {
      headers: { 
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json"
      }
    });
    
    console.log("Statistics response:", res.status);
    
    if (res.ok) {
      const data = await res.json();
      setStatistics(data);
    } else {
      console.error("Failed to fetch statistics:", await res.text());
    }
  } catch (err) {
    console.error("Error fetching statistics:", err);
  } finally {
    setLoading(false);
  }
};

  const fetchRecentActivities = async () => {
  try {
    const token = localStorage.getItem("access_token");
    const res = await fetch(`${API_URL}/api/admin/users/pending`, {
      headers: { 
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json"
      }
    });
    
    console.log("Pending users response:", res.status);
    
    if (res.ok) {
      const data = await res.json();
      console.log("Pending users data:", data);
      
      const activities = data.pending_requests?.slice(0, 5).map(req => ({
        id: req.id,
        user: `${req.first_name} ${req.last_name}`,
        action: `Requested access to ${req.requested_provincial_access || 'N/A'}`,
        time: `${req.days_pending || 0} days ago`,
        type: "registration",
        status: "pending"
      })) || [];
      setRecentActivities(activities);
    }
  } catch (err) {
    console.error("Error fetching recent activities:", err);
  }
};

  useEffect(() => {
    fetchStatistics();
    fetchRecentActivities();
  }, []);

  // Mock data for charts (you can replace with real data later)
  const activityData = [
    { name: "Mon", users: 45, registrations: 3 },
    { name: "Tue", users: 52, registrations: 5 },
    { name: "Wed", users: 48, registrations: 2 },
    { name: "Thu", users: 65, registrations: 8 },
    { name: "Fri", users: 58, registrations: 4 },
    { name: "Sat", users: 32, registrations: 1 },
    { name: "Sun", users: 28, registrations: 2 },
  ];

  const userAccessDistribution = statistics ? [
    { name: "Full Access", value: statistics.users?.with_full_access || 0, color: "#00519C" },
    { name: "Partial Access", value: statistics.users?.with_partial_access || 0, color: "#F7C800" },
    { name: "No Access", value: statistics.users?.no_access || 0, color: "#D12D28" },
  ] : [];

  const registrationStats = statistics ? [
    { name: "Pending", value: statistics.registrations?.pending || 0, color: "#F7C800" },
    { name: "Approved", value: statistics.registrations?.approved || 0, color: "#00519C" },
    { name: "Rejected", value: statistics.registrations?.rejected || 0, color: "#D12D28" },
  ] : [];

  const stats = [
    {
      title: "Total Users",
      value: statistics?.users?.total || 0,
      change: "+12%",
      icon: FiUsers,
      color: "bg-[#00519C]",
      onClick: () => navigate("/admin/users")
    },
    {
      title: "Pending Requests",
      value: statistics?.registrations?.pending || 0,
      change: `${statistics?.registrations?.recent_7_days || 0} this week`,
      icon: FiClock,
      color: "bg-[#F7C800]",
      onClick: () => navigate("/admin/users")
    },
    {
      title: "Active Users",
      value: statistics?.users?.with_full_access || 0,
      change: `${Math.round((statistics?.users?.with_full_access / statistics?.users?.total) * 100) || 0}% of total`,
      icon: FiCheckCircle,
      color: "bg-[#00A86B]",
      onClick: () => navigate("/admin/users")
    },
    {
      title: "Total Uploads",
      value: "2,456",
      change: "+23%",
      icon: FiFileText,
      color: "bg-[#4B1F60]",
    },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#00519C]"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <div
            key={index}
            className="bg-white rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow cursor-pointer"
            onClick={stat.onClick}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-500 text-sm font-medium">
                  {stat.title}
                </p>
                <h3 className="text-2xl font-bold text-gray-800 mt-1">
                  {stat.value}
                </h3>
                <p className="text-sm mt-2 text-gray-500">
                  {stat.change}
                </p>
              </div>
              <div className={`${stat.color} p-3 rounded-lg`}>
                <stat.icon className="text-white text-2xl" />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Activity Chart */}
        <div className="bg-white rounded-lg shadow-sm p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Weekly Activity
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={activityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F0F0F0" />
              <XAxis dataKey="name" tick={{ fill: "#666" }} />
              <YAxis tick={{ fill: "#666" }} />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="users"
                stackId="1"
                stroke="#00519C"
                fill="#00519C"
                fillOpacity={0.6}
              />
              <Area
                type="monotone"
                dataKey="registrations"
                stackId="1"
                stroke="#F7C800"
                fill="#F7C800"
                fillOpacity={0.6}
              />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex items-center justify-center space-x-6 mt-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-[#00519C] rounded-full"></div>
              <span className="text-sm text-gray-600">Active Users</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-[#F7C800] rounded-full"></div>
              <span className="text-sm text-gray-600">New Registrations</span>
            </div>
          </div>
        </div>

        {/* User Access Distribution */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            User Access Status
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={userAccessDistribution}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
              >
                {userAccessDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="space-y-2 mt-4">
            {userAccessDistribution.map((item, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  ></div>
                  <span className="text-sm text-gray-600">{item.name}</span>
                </div>
                <span className="text-sm font-medium text-gray-800">
                  {item.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Activity and Registration Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Activities */}
        <div className="bg-white rounded-lg shadow-sm p-6 lg:col-span-2">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">
              Recent Registration Requests
            </h3>
            <button
              onClick={() => navigate("/admin/users")}
              className="text-sm text-[#00519C] hover:underline"
            >
              View all
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">
                    User
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">
                    Request
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">
                    Time
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody>
                {recentActivities.length > 0 ? (
                  recentActivities.map((activity) => (
                    <tr
                      key={activity.id}
                      className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer"
                      onClick={() => navigate("/admin/users")}
                    >
                      <td className="py-3 px-4">
                        <div className="flex items-center space-x-3">
                          <div className="w-8 h-8 bg-[#00519C] rounded-full flex items-center justify-center">
                            <span className="text-white text-xs font-medium">
                              {activity.user
                                .split(" ")
                                .map((n) => n[0])
                                .join("")}
                            </span>
                          </div>
                          <span className="text-sm font-medium text-gray-800">
                            {activity.user}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-600">
                        {activity.action}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-500">
                        {activity.time}
                      </td>
                      <td className="py-3 px-4">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                          <FiClock className="mr-1" size={10} />
                          {activity.status}
                        </span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="4" className="py-8 text-center text-gray-500">
                      No pending registration requests
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Registration Stats */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Registration Statistics
          </h3>
          <div className="space-y-4">
            {registrationStats.map((stat, index) => (
              <div key={index}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-gray-700">{stat.name}</span>
                  <span className="text-sm font-bold text-gray-900">{stat.value}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="h-2 rounded-full transition-all duration-500"
                    style={{ 
                      width: `${(stat.value / (statistics?.registrations?.total || 1)) * 100}%`,
                      backgroundColor: stat.color 
                    }}
                  ></div>
                </div>
              </div>
            ))}
            
            <div className="pt-4 mt-4 border-t border-gray-200">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Total Requests</span>
                <span className="text-lg font-bold text-gray-900">
                  {statistics?.registrations?.total || 0}
                </span>
              </div>
              <div className="flex justify-between items-center mt-2">
                <span className="text-sm text-gray-600">Last 7 days</span>
                <span className="text-sm font-medium text-[#00519C]">
                  +{statistics?.registrations?.recent_7_days || 0} new
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm p-6">
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            Server Status
          </h4>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600">
              All systems operational
            </span>
          </div>
          <div className="mt-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600">CPU Usage</span>
              <span className="font-medium text-gray-800">45%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-[#00519C] h-2 rounded-full"
                style={{ width: "45%" }}
              ></div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6">
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            Storage Used
          </h4>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-bold text-gray-800">67.8</span>
            <span className="text-sm text-gray-600">GB of 100 GB</span>
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-[#F7C800] h-2 rounded-full"
                style={{ width: "67.8%" }}
              ></div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm p-6">
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            Database Connections
          </h4>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-bold text-gray-800">
              {statistics?.users?.with_full_access || 0}
            </span>
            <span className="text-sm text-gray-600">active users</span>
          </div>
          <div className="mt-2">
            <span className="text-xs text-gray-500">
              {statistics?.users?.with_partial_access || 0} with partial access
            </span>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Quick Actions
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            onClick={() => navigate("/admin/users")}
            className="p-4 bg-[#00519C] text-white rounded-lg hover:bg-[#003d7a] transition-colors"
          >
            <FiUsers className="text-2xl mb-2" />
            <p className="font-medium">Manage Users</p>
            <p className="text-sm opacity-90">
              {statistics?.registrations?.pending || 0} pending requests
            </p>
          </button>
          
          <button
            onClick={() => navigate("/admin/data-upload")}
            className="p-4 bg-[#4B1F60] text-white rounded-lg hover:bg-[#3a1849] transition-colors"
          >
            <FiFileText className="text-2xl mb-2" />
            <p className="font-medium">Upload Data</p>
            <p className="text-sm opacity-90">Manage GIS files</p>
          </button>
          
          <button
            className="p-4 bg-[#F7C800] text-gray-800 rounded-lg hover:bg-[#e6bb00] transition-colors"
          >
            <FiActivity className="text-2xl mb-2" />
            <p className="font-medium">View Reports</p>
            <p className="text-sm opacity-90">System analytics</p>
          </button>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;