import React, { useState } from "react";
import "../css_files/check_animation.css";

const CheckModal = () => {
  const [animate, setAnimate] = useState(false);

  const restartAnimation = () => {
    setAnimate(false);
    setTimeout(() => {
      setAnimate(true);
    }, 10);
  };

  return (
    <div>
      <div className={`success-checkmark ${animate ? "animate" : ""}`}>
        <div className="check-icon">
          <span className="icon-line line-tip"></span>
          <span className="icon-line line-long"></span>
          <div className="icon-circle"></div>
          <div className="icon-fix"></div>
        </div>
      </div>
    </div>
  );
};

export default CheckModal;
