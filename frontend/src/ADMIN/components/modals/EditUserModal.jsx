import React, { useState, useEffect } from "react";
import "../css_files/animation.css";
import { ArrowLeft, ChevronDown } from "lucide-react";
import SuccessModal from "./sucess_modal";
import { API_URL } from "../../../config";

const UserModal = ({ isVisible, onClose, user = null, onSave }) => {
  if (!isVisible) return null;

  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [modalClasses, setModalClasses] = useState(
    isVisible ? "slide-in" : "slide-out"
  );
  const [errors, setErrors] = useState({});
  const [provinces, setProvinces] = useState([]);
  const [municipalitiesByProvince, setMunicipalitiesByProvince] = useState({});
  const [municipalities, setMunicipalities] = useState([]);

  const isEditMode = !!user;

  const [formData, setFormData] = useState({
    name: "",
    firstName: "",
    lastName: "",
    email: "",
    contactNo: "",
    provinceAccess: "",
    municipalAccess: "",
  });

  // 🔹 Fetch provinces + municipalities
  useEffect(() => {
    const fetchLocations = async () => {
      try {
        const token = localStorage.getItem("access_token");
        const res = await fetch(`${API_URL}/api/admin/locations`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (res.ok) {
          const data = await res.json();
          setProvinces(data.provinces || []);
          setMunicipalitiesByProvince(data.municipalities || {});
        } else {
          console.error("Failed to fetch locations:", res.status);
        }
      } catch (err) {
        console.error("Error fetching locations:", err);
      }
    };

    if (isVisible) {
      fetchLocations();
    }
  }, [isVisible]);

  // 🔹 Update municipalities when province changes
  useEffect(() => {
    if (formData.provinceAccess && municipalitiesByProvince[formData.provinceAccess]) {
      setMunicipalities(municipalitiesByProvince[formData.provinceAccess]);
    } else {
      setMunicipalities([]);
    }
  }, [formData.provinceAccess, municipalitiesByProvince]);

  // 🔹 Populate form in edit mode
  useEffect(() => {
    if (user) {
      setFormData({
        name: user.user_name || user.username || "",
        firstName: user.first_name || "",
        lastName: user.last_name || "",
        email: user.email || "",
        contactNo: user.contact_number || user.contact_no || "",
        provinceAccess:
          user.provincial_access || user.requested_provincial_access || "",
        municipalAccess:
          user.municipal_access || user.requested_municipal_access || "",
      });
    }
  }, [user]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });

    if (name === "provinceAccess") {
      setFormData((prev) => ({ ...prev, municipalAccess: "" }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    if (!formData.firstName.trim()) newErrors.firstName = "First name is required";
    if (!formData.lastName.trim()) newErrors.lastName = "Last name is required";
    if (!formData.email.trim()) {
      newErrors.email = "Email is required";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = "Invalid email format";
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      onSave({
        first_name: formData.firstName,
        last_name: formData.lastName,
        email: formData.email,
        contact_number: formData.contactNo,
        provincial_access: formData.provinceAccess || null,
        municipal_access: formData.municipalAccess || null,
      });
      openSuccessModal();
    }
  };

  const openSuccessModal = () => setShowSuccessModal(true);

  const closeSuccessModal = () => {
    setModalClasses("slide-out");
    setTimeout(() => {
      onClose();
      setShowSuccessModal(false);
    }, 490);
  };

  const handleModalBack = () => {
    setModalClasses("slide-out");
    setTimeout(() => onClose(), 490);
  };

  return (
    <>
      <div
        className={`fixed inset-0 bg-black bg-opacity-25 flex justify-end z-50 ${
          isVisible ? "opacity-100" : "opacity-0"
        }`}
        onClick={handleModalBack}
      >
        <div
          className={`${modalClasses} flex flex-wrap justify-start px-14 py-12 md:w-5/12 bg-white rounded-s-lg overflow-y-auto no-scrollbar`}
          onClick={(e) => e.stopPropagation()}
        >
          <form onSubmit={handleSubmit} className="flex-grow">
            <div
              onClick={handleModalBack}
              className="flex cursor-pointer hover:text-[#00519C] mb-4"
            >
              <ArrowLeft size={24} />
              <span className="font-semibold ml-2">Back</span>
            </div>

            <h2 className="text-2xl font-bold mb-2">
              {isEditMode ? "Edit User" : "Add New User"}
            </h2>
            <p className="text-gray-600 mb-6">
              {isEditMode
                ? "Update user details"
                : "Fill out the form to add a user"}
            </p>

            {/* Username (read-only in edit) */}
            {isEditMode && (
              <div className="mb-4">
                <label className="block font-semibold text-gray-700">
                  Username
                </label>
                <input
                  value={formData.name}
                  disabled
                  className="w-full border-2 rounded-lg px-3 py-2 bg-gray-100 border-gray-300"
                />
              </div>
            )}

            {/* First Name */}
            <div className="mb-4">
              <label className="block font-semibold text-gray-700">
                First Name <span className="text-red-500">*</span>
              </label>
              <input
                name="firstName"
                value={formData.firstName}
                onChange={handleChange}
                className={`w-full border-2 rounded-lg px-3 py-2 ${
                  errors.firstName ? "border-red-500" : "border-gray-300"
                }`}
              />
              {errors.firstName && (
                <p className="text-red-500 text-sm">{errors.firstName}</p>
              )}
            </div>

            {/* Last Name */}
            <div className="mb-4">
              <label className="block font-semibold text-gray-700">
                Last Name <span className="text-red-500">*</span>
              </label>
              <input
                name="lastName"
                value={formData.lastName}
                onChange={handleChange}
                className={`w-full border-2 rounded-lg px-3 py-2 ${
                  errors.lastName ? "border-red-500" : "border-gray-300"
                }`}
              />
              {errors.lastName && (
                <p className="text-red-500 text-sm">{errors.lastName}</p>
              )}
            </div>

            {/* Email */}
            <div className="mb-4">
              <label className="block font-semibold text-gray-700">
                Email <span className="text-red-500">*</span>
              </label>
              <input
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                className={`w-full border-2 rounded-lg px-3 py-2 ${
                  errors.email ? "border-red-500" : "border-gray-300"
                }`}
              />
              {errors.email && (
                <p className="text-red-500 text-sm">{errors.email}</p>
              )}
            </div>

            {/* Contact */}
            <div className="mb-4">
              <label className="block font-semibold text-gray-700">
                Contact Number
              </label>
              <input
                name="contactNo"
                value={formData.contactNo}
                onChange={handleChange}
                className="w-full border-2 rounded-lg px-3 py-2 border-gray-300"
                placeholder="09XXXXXXXXX"
              />
            </div>

            {/* Province */}
            <div className="mb-4">
              <label className="block font-semibold text-gray-700">
                Province Access
              </label>
              <div className="relative">
                <select
                  name="provinceAccess"
                  value={formData.provinceAccess}
                  onChange={handleChange}
                  className="w-full border-2 rounded-lg px-3 py-2 appearance-none"
                >
                  <option value="">No Access</option>
                  {provinces.map((p) => (
                    <option key={p.code} value={p.code}>
                      {p.name}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" />
              </div>
            </div>

            {/* Municipality */}
            <div className="mb-6">
              <label className="block font-semibold text-gray-700">
                Municipal Access
              </label>
              <div className="relative">
                <select
  name="municipalAccess"
  value={formData.municipalAccess}
  onChange={handleChange}
  className="w-full border-2 rounded-lg px-3 py-2 appearance-none"
  disabled={!formData.provinceAccess}
>
  <option value="">No Access</option>
  <option value="ALL">All Municipalities</option> {/* ✅ Added */}
  {municipalities.map((m) => (
    <option key={m.code} value={m.code}>
      {m.name}
    </option>
  ))}
</select>

                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" />
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-end">
              <button
                type="button"
                onClick={handleModalBack}
                className="px-6 py-2 border rounded-lg mr-2 hover:bg-gray-100"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-6 py-2 bg-[#00519C] text-white rounded-lg hover:bg-[#003d7a]"
              >
                {isEditMode ? "Save Changes" : "Add User"}
              </button>
            </div>
          </form>
        </div>
      </div>

      <SuccessModal
        isVisible={showSuccessModal}
        onClose={closeSuccessModal}
        text1={isEditMode ? "Successfully updated!" : "Successfully added!"}
        text2={
          isEditMode
            ? "User details have been updated"
            : "New user has been added"
        }
      />
    </>
  );
};

export default UserModal;
