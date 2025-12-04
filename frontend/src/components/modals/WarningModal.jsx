// components/WarningModal.jsx
import React, { useEffect, useState } from "react";
import "./css/warning_animation.css";
import WarningAnimation from "./WarningAnimation.jsx";

const WarningModal = ({
  isVisible,
  onClose,
  title,
  message,
  buttonText = "Ok",
  severity = "warning",
}) => {
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    if (isVisible) {
      setTimeout(() => setAnimate(true), 20); // trigger animation
    } else {
      setAnimate(false);
    }
  }, [isVisible]);

  if (!isVisible) return null;

  const handleClose = () => {
    setAnimate(false);
    setTimeout(() => onClose(), 300); // wait for animation
  };

  const buttonColor =
    severity === "error"
      ? "bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800"
      : "bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-600 hover:to-yellow-700";

  return (
    <div
      className={`fixed inset-0 flex items-center justify-center z-50 backdrop-blur-sm transition-opacity duration-300 ${
        animate ? "opacity-100 bg-black/40" : "opacity-0 bg-black/0"
      }`}
      onClick={handleClose}
    >
      <div
        className={`transform transition-all duration-300 ${
          animate
            ? "scale-100 translate-y-0 opacity-100"
            : "scale-90 translate-y-6 opacity-0"
        } bg-white rounded-2xl shadow-2xl w-[28em] max-w-full mx-4`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex flex-col items-center px-6 py-8">
          {/* Animated Icon */}
          <WarningAnimation severity={severity} />

          {/* Title */}
          <h2 className="mt-4 text-2xl font-bold text-center text-gray-800">
            {title}
          </h2>

          {/* Message */}
          <p className="mt-2 text-base text-center text-gray-600 leading-relaxed">
            {message}
          </p>

          {/* Button */}
          <button
            onClick={handleClose}
            className={`${buttonColor} mt-8 px-10 py-3 rounded-lg text-white font-semibold shadow-md transition-transform duration-200 hover:scale-105 active:scale-95`}
          >
            {buttonText}
          </button>
        </div>
      </div>
    </div>
  );
};

export default WarningModal;
