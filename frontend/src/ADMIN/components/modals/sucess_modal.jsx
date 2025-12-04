import React, { useState } from "react";
import "../css_files/animation.css";
import CheckModal from "./check_animation.jsx";

const Modals = ({ isVisible, onClose, text1, text2 }) => {
  if (!isVisible) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
  };

  const [modalClasses, setModalClasses] = useState(
    isVisible ? "pop-in" : "pop-out"
  );
  const handleModalBack = () => {
    setModalClasses("pop-out");
    setTimeout(() => onClose(), 450);
  };
  return (
    <div
      className={`fixed inset-0 bg-black bg-opacity-25 flex items-center justify-center ${
        isVisible ? "opacity-100" : "opacity-0"
      }`}
      onClick={handleModalBack}
    >
      <div
        className={`${modalClasses} w-[28em] h-[22em] bg-white rounded-2xl`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className=" flex items-center flex-col pt-8">
          <div className="flex">
            <CheckModal></CheckModal>
          </div>
          <div className="flex justify-center items-center pt-2">
            <p className=" font-bold text-2xl text-center mx-16">
              {text1 || "No text input"}
            </p>
          </div>
          <div className="pt-1">
            <p className=" font-medium text-base text-left mx-20 pt-1">
              {text2 || "No text input"}
            </p>
          </div>
          <div className="flex flex-row pt-7">
            <div className="">
              <button
                type="submit"
                onClick={handleModalBack}
                className="bg-green-800 hover:bg-green-700 font-semibold rounded-lg px-10 py-3 text-white text-base"
              >
                Ok
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Modals;
