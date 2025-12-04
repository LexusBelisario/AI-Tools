import React, { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  FiHome,
  FiSettings,
  FiUsers,
  FiShield,
  FiUserCheck,
  FiDatabase,
  FiUpload,
  FiFolder,
  FiChevronDown,
  FiMenu,
} from "react-icons/fi";
import BLGF_LOGO from "../images/BLGF_Logo.png";

const AdminSidebar = ({ isOpen, toggleSidebar }) => {
  const [userSettingsOpen, setUserSettingsOpen] = useState(true);
  const [dataManagementOpen, setDataManagementOpen] = useState(true);

  const menuItems = [
    {
      title: "DASHBOARD",
      icon: FiHome,
      path: "/admin/dashboard",
      type: "single",
    },
    {
      title: "USER SETTINGS",
      icon: FiSettings,
      type: "dropdown",
      isOpen: userSettingsOpen,
      toggle: () => setUserSettingsOpen(!userSettingsOpen),
      subItems: [
        {
          title: "User Management",
          path: "/admin/users",
          icon: FiUsers,
        },
        {
          title: "Role Management",
          path: "/admin/role-management",
          icon: FiShield,
        },
        { title: "User Groups", path: "/admin/user-groups", icon: FiUserCheck },
      ],
    },
    {
      title: "DATA MANAGEMENT",
      icon: FiDatabase,
      type: "dropdown",
      isOpen: dataManagementOpen,
      toggle: () => setDataManagementOpen(!dataManagementOpen),
      subItems: [
        {
          title: "CSV/SHP File Upload",
          path: "/admin/data-upload",
          icon: FiUpload,
        },
        { title: "Data Manager", path: "/admin/data-manager", icon: FiFolder },
      ],
    },
  ];

  return (
    <div
      className={`fixed left-0 top-0 h-full bg-white border-r border-[#E0E0E0] transition-all duration-300 z-40 ${
        isOpen ? "w-64" : "w-16"
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-[#E0E0E0]">
        <div className="flex items-center space-x-3">
          <img
            src={BLGF_LOGO}
            alt="BLGF Logo"
            className="w-10 h-10 object-contain"
          />
          {isOpen && (
            <div>
              <h2 className="text-lg font-extrabold text-[#222222]">
                LGU WEB-APP
              </h2>
              <p className="text-xs font-bold text-[#777777]">Admin Control</p>
            </div>
          )}
        </div>
        <button
          onClick={toggleSidebar}
          className="p-2 hover:bg-gray-100 rounded-md transition-colors"
        >
          <FiMenu className="text-gray-600" />
        </button>
      </div>

      {/* Navigation */}
      <nav className="p-4">
        {menuItems.map((item, index) => (
          <div key={index} className="mb-2">
            {item.type === "single" ? (
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-all ${
                    isActive
                      ? "bg-[#4B1F60] text-white"
                      : "text-gray-700 hover:bg-gray-50"
                  }`
                }
              >
                <item.icon className="text-lg" />
                {isOpen && (
                  <span className="text-sm font-bold">{item.title}</span>
                )}
              </NavLink>
            ) : (
              <>
                <button
                  onClick={item.toggle}
                  className="w-full flex items-center justify-between px-3 py-2.5 text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
                >
                  <div className="flex items-center space-x-3">
                    <item.icon className="text-lg" />
                    {isOpen && (
                      <span className="text-sm font-bold">{item.title}</span>
                    )}
                  </div>
                  {isOpen && (
                    <FiChevronDown
                      className={`text-gray-400 transition-transform ${
                        item.isOpen ? "rotate-180" : ""
                      }`}
                    />
                  )}
                </button>

                {isOpen && item.isOpen && (
                  <div className="mt-1 ml-3 pl-4 border-l-2 border-[#E0E0E0]">
                    {item.subItems.map((subItem, subIndex) => (
                      <NavLink
                        key={subIndex}
                        to={subItem.path}
                        className={({ isActive }) =>
                          `flex items-center space-x-2 px-3 py-2 rounded-lg text-sm transition-all ${
                            isActive
                              ? "bg-[#4B1F60] text-white"
                              : "text-gray-600 hover:bg-gray-50"
                          }`
                        }
                      >
                        <subItem.icon className="text-base" />
                        <span>{subItem.title}</span>
                      </NavLink>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        ))}
      </nav>
    </div>
  );
};

export default AdminSidebar;
