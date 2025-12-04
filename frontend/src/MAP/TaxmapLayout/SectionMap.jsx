import React from "react";
import Form from "./Form.jsx";
import { useSchema } from "../SchemaContext.jsx";

const SectionMap = ({ onApply, onClose }) => {
  const { schema } = useSchema();

  const handleApply = (payload) => {
    // Extract barangay string
    const barangay = payload.barangay;

    onApply({
      ...payload,
      barangay,
      schema
    });
  };

  return (
    <div className="sectionmap-wrapper">
      <Form
        embedded={true}
        showApply={true}
        onApply={handleApply}
        onClose={onClose}
      />
    </div>
  );
};

export default SectionMap;
