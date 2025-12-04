import { useState, useEffect } from "react";
import LoadingOverlay from "../LOADINGSCREEN/LoadingOverlay.jsx";

function LoadingHandler() {
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    window.setLoadingProgress = (state) => {
      setLoading(state);
    };
  }, []);

  return <LoadingOverlay visible={loading} />;
}

export default LoadingHandler;
