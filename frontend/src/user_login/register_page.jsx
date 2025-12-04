import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { API_URL } from "../config";
import {
  FiUser,
  FiMail,
  FiLock,
  FiPhone,
  FiEye,
  FiEyeOff,
} from "react-icons/fi";

function RegisterPage() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: "",
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    contactNo: "",
    tentativeProvince: "",
    tentativeMunicipal: "",
  });
  const [provinces, setProvinces] = useState([]);
  const [municipalitiesByProvince, setMunicipalitiesByProvince] = useState({});
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [passwordError, setPasswordError] = useState("");

  useEffect(() => {
    const fetchLocations = async () => {
      try {
        const res = await fetch(`${API_URL}/api/admin/locations`);
        if (res.ok) {
          const data = await res.json();
          setProvinces(data.provinces || []);
          setMunicipalitiesByProvince(data.municipalities || {});
        } else {
          console.error(
            "Failed to fetch locations:",
            res.status,
            res.statusText
          );
        }
      } catch (err) {
        console.error("Failed to load locations:", err);
      }
    };
    fetchLocations();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });

    if (name === "tentativeProvince") {
      setFormData((prev) => ({ ...prev, tentativeMunicipal: "" }));
    }

    if (name === "confirmPassword" || name === "password") {
      if (name === "confirmPassword") {
        setPasswordError(
          value !== formData.password ? "Passwords do not match" : ""
        );
      } else if (name === "password" && formData.confirmPassword) {
        setPasswordError(
          value !== formData.confirmPassword ? "Passwords do not match" : ""
        );
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const provinceObj = provinces.find(
      (p) => p.code === formData.tentativeProvince
    );
    const municipalObj =
      formData.tentativeMunicipal === "ALL"
        ? { code: "ALL", name: "All Municipalities" }
        : municipalitiesByProvince[formData.tentativeProvince]?.find(
            (m) => m.code === formData.tentativeMunicipal
          );

    const requestData = {
      username: formData.username,
      first_name: formData.firstName,
      last_name: formData.lastName,
      email: formData.email,
      password: formData.password,
      contact_number: formData.contactNo,
      requested_provincial_access: provinceObj?.code || "",
      requested_municipal_access: municipalObj?.code || "",
      requested_provincial_code: provinceObj?.code || "",
      requested_municipal_code: municipalObj?.code || "",
    };

    try {
      const res = await fetch(`${API_URL}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData),
      });

      const data = await res.json();
      if (!res.ok) {
        alert(data?.detail || "Registration failed");
        return;
      }

      alert("Registration request submitted! Please wait for admin approval.");
      navigate("/login");
    } catch (err) {
      console.error("Registration error:", err);
      alert("Failed to register. Try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center px-4">
      <div className="w-full max-w-lg bg-white shadow-lg rounded-xl p-8 max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-bold text-center text-[#00519C] mb-6">
          Create an Account
        </h2>
        <form className="space-y-4" onSubmit={handleSubmit}>
          {/* Username */}
          <div>
            <label className="text-sm font-semibold text-gray-700">
              Username
            </label>
            <div className="flex items-center border rounded-lg px-3">
              <FiUser className="text-gray-400 mr-2" />
              <input
                type="text"
                name="username"
                value={formData.username}
                onChange={handleChange}
                className="flex-1 py-2 outline-none"
                required
              />
            </div>
          </div>

          {/* First Name */}
          <div>
            <label className="text-sm font-semibold text-gray-700">
              First Name
            </label>
            <div className="flex items-center border rounded-lg px-3">
              <FiUser className="text-gray-400 mr-2" />
              <input
                type="text"
                name="firstName"
                value={formData.firstName}
                onChange={handleChange}
                className="flex-1 py-2 outline-none"
                required
              />
            </div>
          </div>

          {/* Last Name */}
          <div>
            <label className="text-sm font-semibold text-gray-700">
              Last Name
            </label>
            <div className="flex items-center border rounded-lg px-3">
              <FiUser className="text-gray-400 mr-2" />
              <input
                type="text"
                name="lastName"
                value={formData.lastName}
                onChange={handleChange}
                className="flex-1 py-2 outline-none"
                required
              />
            </div>
          </div>

          {/* Email */}
          <div>
            <label className="text-sm font-semibold text-gray-700">Email</label>
            <div className="flex items-center border rounded-lg px-3">
              <FiMail className="text-gray-400 mr-2" />
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                className="flex-1 py-2 outline-none"
                required
              />
            </div>
          </div>

          {/* Password */}
          <div>
            <label className="text-sm font-semibold text-gray-700">
              Password
            </label>
            <div className="flex items-center border rounded-lg px-3">
              <FiLock className="text-gray-400 mr-2" />
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleChange}
                className="flex-1 py-2 outline-none"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="text-gray-400 hover:text-gray-600 p-1"
              >
                {showPassword ? <FiEyeOff size={20} /> : <FiEye size={20} />}
              </button>
            </div>
          </div>

          {/* Confirm Password */}
          <div>
            <label className="text-sm font-semibold text-gray-700">
              Confirm Password
            </label>
            <div
              className={`flex items-center border rounded-lg px-3 ${
                passwordError ? "border-red-500" : ""
              }`}
            >
              <FiLock className="text-gray-400 mr-2" />
              <input
                type={showConfirmPassword ? "text" : "password"}
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                className="flex-1 py-2 outline-none"
                required
              />
              <button
                type="button"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                className="text-gray-400 hover:text-gray-600 p-1"
              >
                {showConfirmPassword ? (
                  <FiEyeOff size={20} />
                ) : (
                  <FiEye size={20} />
                )}
              </button>
            </div>
            {passwordError && (
              <p className="text-xs text-red-500 mt-1">{passwordError}</p>
            )}
          </div>

          {/* Contact No */}
          <div>
            <label className="text-sm font-semibold text-gray-700">
              Contact Number
            </label>
            <div className="flex items-center border rounded-lg px-3">
              <FiPhone className="text-gray-400 mr-2" />
              <input
                type="text"
                name="contactNo"
                value={formData.contactNo}
                onChange={handleChange}
                className="flex-1 py-2 outline-none"
                placeholder="09XXXXXXXXX"
              />
            </div>
          </div>

          {/* Province Dropdown */}
          <div>
            <label className="text-sm font-semibold text-gray-700">
              Requested Province Access
            </label>
            <select
              name="tentativeProvince"
              value={formData.tentativeProvince}
              onChange={handleChange}
              className="w-full border rounded-lg py-2 px-3 bg-white text-gray-900"
            >
              <option value="">Select Province</option>
              {provinces.map((p) => (
                <option key={p.code} value={p.code}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {/* Municipality Dropdown */}
          <div>
            <label className="text-sm font-semibold text-gray-700">
              Requested Municipality Access
            </label>
            <select
              name="tentativeMunicipal"
              value={formData.tentativeMunicipal}
              onChange={handleChange}
              className="w-full border rounded-lg py-2 px-3 bg-white text-gray-900"
              disabled={!formData.tentativeProvince}
            >
              <option value="">Select Municipality</option>
              <option value="ALL">All Municipalities</option>
              {municipalitiesByProvince[formData.tentativeProvince]?.map(
                (m) => (
                  <option key={m.code} value={m.code}>
                    {m.name}
                  </option>
                )
              )}
            </select>
          </div>

          <button
            type="submit"
            disabled={loading || passwordError}
            className={`w-full font-bold py-3 rounded-lg transition ${
              loading || passwordError
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-[#00519C] hover:bg-blue-700 text-white"
            }`}
          >
            {loading ? "Submitting..." : "Submit Registration Request"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default RegisterPage;
