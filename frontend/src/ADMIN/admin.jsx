import React, { useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import AdminHeader from "./components/admin_header";
import AdminSidebar from "./components/admin_sidebar";
import AdminDashboard from "./pages/admin_dashboard";
import AdminFileUpload from "./pages/admin_file_upload";
import UserManagement from "./pages/user_management";

const Admin = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="min-h-screen bg-[#F5F5F5] flex">
      {/* Sidebar */}
      <AdminSidebar
        isOpen={sidebarOpen}
        toggleSidebar={() => setSidebarOpen(!sidebarOpen)}
      />

      {/* Main Content */}
      <div
        className={`flex-1 flex flex-col transition-all duration-300 ${
          sidebarOpen ? "ml-64" : "ml-16"
        }`}
      >
        <AdminHeader />

        <main className="flex-1 p-6 bg-[#F5F5F5]">
          <Routes>
            <Route
              path="/"
              element={<Navigate to="/admin/dashboard" replace />}
            />
            <Route path="/dashboard" element={<AdminDashboard />} />
            <Route path="/users" element={<UserManagement />} />
            <Route path="/data-upload" element={<AdminFileUpload />} />{" "}
          </Routes>
        </main>
      </div>
    </div>
  );
};

export default Admin;
