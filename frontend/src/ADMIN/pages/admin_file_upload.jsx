import React, { useState, useRef, useCallback } from "react";
import {
  FiUpload,
  FiFile,
  FiCheckCircle,
  FiXCircle,
  FiAlertCircle,
  FiTrash2,
  FiPause,
  FiPlay,
  FiRefreshCw,
} from "react-icons/fi";

const AdminFileUpload = () => {
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // Mock uploaded files for demonstration
  const [uploadedFiles, setUploadedFiles] = useState([
    {
      id: 1,
      name: "013_Brgy_Bilibiran.shp",
      progress: 100,
      status: "success",
      size: "2.4 MB",
      isPaused: false,
    },
    {
      id: 2,
      name: "014_Brgy_Tagpos.csv",
      progress: 75,
      status: "uploading",
      size: "1.2 MB",
      isPaused: false,
    },
    {
      id: 3,
      name: "013_Brgy_Pantok.shp",
      progress: 100,
      status: "error",
      error: "File size exceeds limit",
      size: "15.8 MB",
      isPaused: false,
    },
    {
      id: 4,
      name: "013_Brgy_Bilibiran.shp",
      progress: 100,
      status: "success",
      size: "2.4 MB",
      isPaused: false,
    },
  ]);

  // Handle drag events
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  }, []);

  // Handle file selection
  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    handleFiles(selectedFiles);
  };

  const handleFiles = (fileList) => {
    // Filter for allowed file types
    const allowedTypes = [
      ".shp",
      ".csv",
      ".dbf",
      ".shx",
      ".prj",
      ".geojson",
      ".kml",
    ];
    const validFiles = fileList.filter((file) => {
      const extension = "." + file.name.split(".").pop().toLowerCase();
      return allowedTypes.includes(extension);
    });

    if (validFiles.length !== fileList.length) {
      alert(
        "Some files were skipped. Only SHP, CSV, DBF, SHX, PRJ, GeoJSON, and KML files are allowed."
      );
    }

    // Add files to upload queue
    const newFiles = validFiles.map((file) => ({
      id: Date.now() + Math.random(),
      name: file.name,
      size: formatFileSize(file.size),
      progress: 0,
      status: "pending",
      file: file,
      isPaused: false,
    }));

    setUploadedFiles((prev) => [...newFiles, ...prev]);

    // TODO: Implement actual file upload logic here
    uploadFiles(newFiles);
  };

  const uploadFiles = async (filesToUpload) => {
    // Placeholder for backend upload logic
    for (const fileInfo of filesToUpload) {
      try {
        // Simulate upload progress
        await simulateUpload(fileInfo.id);
      } catch (error) {
        updateFileStatus(fileInfo.id, "error", error.message);
      }
    }
  };

  // Simulate upload progress (remove when implementing real upload)
  const simulateUpload = async (fileId) => {
    updateFileStatus(fileId, "uploading");

    for (let i = 0; i <= 100; i += 10) {
      await new Promise((resolve) => setTimeout(resolve, 200));

      // Check if paused
      const file = uploadedFiles.find((f) => f.id === fileId);
      if (file?.isPaused) {
        await new Promise((resolve) => {
          const checkPause = setInterval(() => {
            const currentFile = uploadedFiles.find((f) => f.id === fileId);
            if (!currentFile?.isPaused) {
              clearInterval(checkPause);
              resolve();
            }
          }, 100);
        });
      }

      updateFileProgress(fileId, i);
    }

    // Randomly simulate success or failure
    if (Math.random() > 0.8) {
      updateFileStatus(fileId, "error", "Upload failed");
    } else {
      updateFileStatus(fileId, "success");
    }
  };

  const updateFileProgress = (fileId, progress) => {
    setUploadedFiles((prev) =>
      prev.map((file) => (file.id === fileId ? { ...file, progress } : file))
    );
  };

  const updateFileStatus = (fileId, status, error = null) => {
    setUploadedFiles((prev) =>
      prev.map((file) =>
        file.id === fileId ? { ...file, status, error } : file
      )
    );
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const getProgressBarColor = (status) => {
    switch (status) {
      case "success":
        return "bg-[#0BDA51]"; // Changed from #F7C800 to green
      case "error":
        return "bg-[#D12D28]";
      case "uploading":
        return "bg-[#00519C]";
      default:
        return "bg-gray-300";
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "success":
        return <FiCheckCircle className="text-[#0BDA51] w-5 h-5" />; // Updated to match progress bar color
      case "error":
        return <FiXCircle className="text-[#D12D28] w-5 h-5" />;
      case "uploading":
        return (
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-[#00519C]" />
        );
      default:
        return <FiFile className="text-gray-400 w-5 h-5" />;
    }
  };

  const handleDelete = (fileId) => {
    setUploadedFiles((prev) => prev.filter((file) => file.id !== fileId));
  };

  const handlePauseResume = (fileId) => {
    setUploadedFiles((prev) =>
      prev.map((file) =>
        file.id === fileId ? { ...file, isPaused: !file.isPaused } : file
      )
    );
  };

  const handleRetry = (fileId) => {
    const file = uploadedFiles.find((f) => f.id === fileId);
    if (file) {
      updateFileStatus(fileId, "pending");
      updateFileProgress(fileId, 0);
      uploadFiles([file]);
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="bg-white rounded-lg shadow-sm p-8">
        <h2 className="text-2xl font-semibold text-gray-800 mb-2">
          Upload your files here
        </h2>
        <p className="text-gray-600 mb-8">Upload your shp/csv file here</p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Area - Left Side */}
          <div
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
              isDragging
                ? "border-[#00519C] bg-blue-50"
                : "border-gray-400 hover:border-[#00519C]"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <FiUpload className="mx-auto text-6xl text-gray-400 mb-4" />

            <p className="text-lg font-medium text-gray-700 mb-2">
              Drag and drop your files here
            </p>
            <p className="text-gray-500 mb-6">-or-</p>

            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-8 py-3 bg-[#F7C800] text-gray-800 font-medium rounded-lg hover:bg-yellow-500 transition-colors"
            >
              Browse Files
            </button>

            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".shp,.csv,.dbf,.shx,.prj,.geojson,.kml"
              onChange={handleFileSelect}
              className="hidden"
            />

            <p className="text-sm text-gray-500 mt-4">
              Supported formats: SHP, CSV, DBF, SHX, PRJ, GeoJSON, KML
            </p>
          </div>

          {/* File List - Right Side */}
          <div className="flex flex-col">
            {uploadedFiles.length > 0 ? (
              <>
                <h3 className="text-lg font-semibold text-gray-800 mb-4">
                  Uploaded Files
                </h3>
                <div className="space-y-4 overflow-y-auto max-h-[500px] pr-2">
                  {uploadedFiles.map((file) => (
                    <div key={file.id} className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3 flex-1">
                          {getStatusIcon(file.status)}
                          <span className="font-medium text-gray-800">
                            {file.name}
                          </span>
                          <span className="text-sm text-gray-500">
                            {file.size}
                          </span>
                        </div>

                        <div className="flex items-center space-x-2">
                          {file.status === "uploading" && (
                            <>
                              <span className="text-sm text-gray-600 mr-2">
                                {file.progress}%
                              </span>
                              <button
                                onClick={() => handlePauseResume(file.id)}
                                className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                                title={file.isPaused ? "Resume" : "Pause"}
                              >
                                {file.isPaused ? (
                                  <FiPlay className="w-4 h-4" />
                                ) : (
                                  <FiPause className="w-4 h-4" />
                                )}
                              </button>
                            </>
                          )}

                          {file.status === "error" && (
                            <button
                              onClick={() => handleRetry(file.id)}
                              className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                              title="Retry"
                            >
                              <FiRefreshCw className="w-4 h-4" />
                            </button>
                          )}

                          <button
                            onClick={() => handleDelete(file.id)}
                            className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                            title="Delete"
                          >
                            <FiTrash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>

                      <div className="ml-8">
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full transition-all duration-300 ${getProgressBarColor(
                              file.status
                            )}`}
                            style={{ width: `${file.progress}%` }}
                          />
                        </div>

                        {file.error && (
                          <p className="mt-2 text-sm text-[#D12D28]">
                            {file.error}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-gray-400">
                  <FiFile className="mx-auto text-5xl mb-3" />
                  <p className="text-lg">No files uploaded yet</p>
                  <p className="text-sm mt-1">Upload files to see them here</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminFileUpload;
