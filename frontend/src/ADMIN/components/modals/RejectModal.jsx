import React, { useState, useEffect } from "react";
import "../css_files/animation.css";
import { ArrowLeft } from "lucide-react";

const RejectModal = ({ isVisible, onClose, onConfirm }) => {
  const [modalClasses, setModalClasses] = useState("slide-out");
  const [selectedReason, setSelectedReason] = useState("");
  const [customReason, setCustomReason] = useState("");

  useEffect(() => {
    if (isVisible) {
      setModalClasses("slide-in");
    } else {
      setModalClasses("slide-out");
    }
  }, [isVisible]);

  if (!isVisible) return null;

  const rejectionReasons = [
    "Province or Municipality chosen by the user is not available in the system",
    "Duplicate account found",
    "Invalid or incomplete information provided",
    "Suspicious registration attempt",
    "Others"
  ];

  const handleModalBack = () => {
    setModalClasses("slide-out");
    setTimeout(() => {
      setSelectedReason("");
      setCustomReason("");
      onClose();
    }, 490);
  };

  const handleConfirm = () => {
    const reason =
      selectedReason === "Others" ? customReason.trim() : selectedReason;
    if (!reason) return;
    onConfirm(reason);
    handleModalBack();
  };

  // Determine if the custom input should be active
  const isOthersSelected = selectedReason === "Others";

  return (
    <div
      className={`fixed inset-0 bg-black bg-opacity-25 flex justify-end z-50`}
      onClick={handleModalBack}
    >
      <div
        className={`${modalClasses} flex flex-col px-10 py-8 md:w-5/12 bg-white rounded-s-lg`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          onClick={handleModalBack}
          className="flex cursor-pointer hover:text-[#00519C] mb-4"
        >
          <ArrowLeft size={24} />
          <span className="font-semibold ml-2">Back</span>
        </div>

        <h2 className="text-2xl font-bold mb-4">Reject Registration</h2>

        {/* Dropdown */}
        <label className="block font-semibold text-gray-700 mb-2">
          Select a Reason
        </label>
        <select
          value={selectedReason}
          onChange={(e) => setSelectedReason(e.target.value)}
          className="w-full border-2 rounded-lg px-3 py-2 mb-4 focus:ring-2 focus:ring-[#00519C] focus:border-[#00519C] outline-none"
        >
          <option value="">-- Choose a reason --</option>
          {rejectionReasons.map((r, i) => (
            <option key={i} value={r}>
              {r}
            </option>
          ))}
        </select>

        {/* Custom Textarea */}
        <label className="block font-semibold text-gray-700 mb-2">
          Other Reason (only if “Others” is selected)
        </label>
        <textarea
          value={customReason}
          onChange={(e) => setCustomReason(e.target.value)}
          rows={3}
          disabled={!isOthersSelected}
          className={`w-full border-2 rounded-lg px-3 py-2 mb-4 outline-none ${
            isOthersSelected
              ? "focus:ring-2 focus:ring-[#00519C] focus:border-[#00519C]"
              : "bg-gray-100 cursor-not-allowed text-gray-500"
          }`}
          placeholder={
            isOthersSelected
              ? "Type custom reason here..."
              : "Select 'Others' to enable this field"
          }
        />

        {/* Actions */}
        <div className="flex justify-end">
          <button
            onClick={handleModalBack}
            className="px-6 py-2 border rounded-lg mr-2 hover:bg-gray-100"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={
              !selectedReason ||
              (selectedReason === "Others" && !customReason.trim())
            }
            className={`px-6 py-2 rounded-lg text-white ${
              !selectedReason ||
              (selectedReason === "Others" && !customReason.trim())
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-red-600 hover:bg-red-700"
            }`}
          >
            Confirm Reject
          </button>
        </div>
      </div>
    </div>
  );
};

export default RejectModal;
