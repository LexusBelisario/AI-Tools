import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { API_URL } from "../config";
import WarningModal from "../components/modals/WarningModal";

function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [usernameFocused, setUsernameFocused] = useState(false);
  const [passwordFocused, setPasswordFocused] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const [accessModal, setAccessModal] = useState({
    isVisible: false,
    title: "",
    message: "",
    severity: "warning",
    canProceed: false,
  });

  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const resp = await fetch(`${API_URL}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await resp.json();
      console.log("Login response:", data);

      if (!resp.ok) {
        if (resp.status === 403) {
          setAccessModal({
            isVisible: true,
            title: "Access Denied",
            message:
              data?.detail ||
              "You do not have municipal access yet. Please contact administrator.",
            severity: "error",
            canProceed: false,
          });
        } else {
          alert(`Login failed: ${data?.detail || "Invalid credentials"}`);
        }
        return;
      }

      const token = data?.access_token;
      localStorage.setItem("accessToken", token);

      if (data?.user_type === "admin") {
        navigate("/admin");
      } else {
        navigate("/map");
      }
    } catch (error) {
      console.error("Login error:", error);
      alert("Login failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleModalClose = () => {
    setAccessModal({
      isVisible: false,
      title: "",
      message: "",
      severity: "warning",
      canProceed: false,
    });
  };

  return (
    <div className="min-h-screen bg-[#F5F5F5] flex items-center justify-center px-4">
      {/* Philippine Map Background */}
      <div className="absolute inset-0 flex items-center justify-center opacity-[0.03]">
        <img
          src="https://www.nicepng.com/png/detail/102-1026494_philippine-map-clipart-png-vice-mayors-league-of.png"
          alt="Philippine Map Background"
          className="w-[90%] h-[90%] object-contain"
        />
      </div>

      {/* Decorative Elements */}
      <div className="absolute top-10 left-10 w-32 h-32 bg-gradient-to-br from-[#F2C300]/10 to-[#D4AF37]/5 rounded-full blur-3xl"></div>
      <div className="absolute bottom-10 right-10 w-40 h-40 bg-gradient-to-br from-[#0B4EA2]/10 to-[#702082]/5 rounded-full blur-3xl"></div>
      <div className="absolute top-1/3 right-20 w-24 h-24 bg-gradient-to-br from-[#C51E2A]/10 to-[#7F0F1A]/5 rounded-full blur-2xl"></div>

      {/* Login Card */}
      <div className="relative bg-white rounded-3xl p-8 shadow-[0_20px_60px_rgba(34,34,34,0.08)] border border-[#222222]/5 w-[500px] min-h-[600px] flex flex-col backdrop-blur-sm">
        {/* Header Section */}
        <div className="flex flex-col items-center justify-center mb-8">
          <div className="relative mb-4">
            <div className="w-20 h-20 bg-gradient-to-br from-[#F2C300] to-[#D4AF37] rounded-full p-[2px] shadow-lg">
              <div className="w-full h-full bg-white rounded-full flex items-center justify-center">
                <div className="text-3xl font-bold bg-gradient-to-br from-[#0B4EA2] to-[#702082] text-transparent bg-clip-text">
                  RPT
                </div>
              </div>
            </div>
            <div className="absolute -top-2 left-1/2 -translate-x-1/2">
              <svg
                className="w-4 h-4 text-[#F2C300]"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
            </div>
            <div className="absolute top-1/2 -left-4 -translate-y-1/2">
              <svg
                className="w-4 h-4 text-[#F2C300]"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
            </div>
            <div className="absolute top-1/2 -right-4 -translate-y-1/2">
              <svg
                className="w-4 h-4 text-[#F2C300]"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
            </div>
          </div>
          <h1 className="text-[#222222] text-4xl font-bold tracking-wide">
            RPTGIS
          </h1>
          <p className="text-[#222222]/60 text-sm mt-2">
            Welcome to Real Property Tax GIS
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="flex-1 flex flex-col justify-center space-y-6 max-w-sm mx-auto w-full"
        >
          {/* Username */}
          <div className="relative group">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 flex items-center pointer-events-none z-10">
                <div
                  className={`w-12 h-full flex items-center justify-center border-r transition-colors duration-200 ${
                    usernameFocused || username
                      ? "border-[#F2C300]/50"
                      : "border-[#222222]/10"
                  }`}
                >
                  <svg
                    className={`w-5 h-5 transition-colors duration-200 ${
                      usernameFocused || username
                        ? "text-[#F2C300]"
                        : "text-[#222222]/40"
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                    />
                  </svg>
                </div>
              </div>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                onFocus={() => setUsernameFocused(true)}
                onBlur={() => setUsernameFocused(false)}
                className={`w-full pl-14 pr-4 py-4 bg-[#F5F5F5] border rounded-xl text-[#222222] placeholder-transparent focus:outline-none transition-all duration-300 ${
                  usernameFocused || username
                    ? "border-[#F2C300] bg-white shadow-[0_0_20px_rgba(242,195,0,0.2)]"
                    : "border-[#222222]/10 hover:border-[#222222]/20"
                }`}
                placeholder="Username"
                id="username"
              />
              <label
                htmlFor="username"
                className={`absolute left-14 transition-all duration-300 pointer-events-none ${
                  usernameFocused || username
                    ? "top-0 -translate-y-1/2 text-xs text-[#F2C300] bg-white px-2 font-medium"
                    : "top-4 text-base text-[#222222]/60"
                }`}
              >
                Username
              </label>
            </div>
          </div>

          {/* Password */}
          <div className="relative group">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 flex items-center pointer-events-none z-10">
                <div
                  className={`w-12 h-full flex items-center justify-center border-r transition-colors duration-200 ${
                    passwordFocused || password
                      ? "border-[#F2C300]/50"
                      : "border-[#222222]/10"
                  }`}
                >
                  <svg
                    className={`w-5 h-5 transition-colors duration-200 ${
                      passwordFocused || password
                        ? "text-[#F2C300]"
                        : "text-[#222222]/40"
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                    />
                  </svg>
                </div>
              </div>
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onFocus={() => setPasswordFocused(true)}
                onBlur={() => setPasswordFocused(false)}
                className={`w-full pl-14 pr-12 py-4 bg-[#F5F5F5] border rounded-xl text-[#222222] placeholder-transparent focus:outline-none transition-all duration-300 ${
                  passwordFocused || password
                    ? "border-[#F2C300] bg-white shadow-[0_0_20px_rgba(242,195,0,0.2)]"
                    : "border-[#222222]/10 hover:border-[#222222]/20"
                }`}
                placeholder="Password"
                id="password"
              />
              <label
                htmlFor="password"
                className={`absolute left-14 transition-all duration-300 pointer-events-none ${
                  passwordFocused || password
                    ? "top-0 -translate-y-1/2 text-xs text-[#F2C300] bg-white px-2 font-medium"
                    : "top-4 text-base text-[#222222]/60"
                }`}
              >
                Password
              </label>
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className={`absolute inset-y-0 right-3 flex items-center transition-all duration-200 z-10 ${
                  passwordFocused || password
                    ? "text-[#F2C300] hover:text-[#D4AF37]"
                    : "text-[#222222]/40 hover:text-[#222222]/60"
                }`}
              >
                {showPassword ? (
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"
                    />
                  </svg>
                ) : (
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                    />
                  </svg>
                )}
              </button>
            </div>
          </div>

          {/* Remember / Forgot */}
          <div className="flex items-center justify-between">
            <label className="flex items-center text-[#222222]/60 text-sm cursor-pointer hover:text-[#222222]/80 transition-colors">
              <input
                type="checkbox"
                className="mr-2 rounded border-[#222222]/20 text-[#F2C300] focus:ring-[#F2C300] focus:ring-offset-0 focus:ring-2"
              />
              Remember me
            </label>
            <a
              href="#"
              className="text-sm text-[#702082] hover:text-[#C51E2A] transition-colors duration-200 font-medium"
            >
              Forgot password?
            </a>
          </div>

          {/* Submit */}
          <div className="pt-2">
            <button
              type="submit"
              disabled={isLoading}
              className={`w-full bg-gradient-to-r from-[#F2C300] to-[#D4AF37] hover:from-[#D4AF37] hover:to-[#F2C300] text-[#222222] text-lg font-bold py-4 rounded-xl transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98] shadow-lg hover:shadow-[0_12px_30px_rgba(242,195,0,0.3)] relative overflow-hidden ${
                isLoading ? "opacity-50 cursor-not-allowed" : ""
              }`}
            >
              <span className="absolute inset-0 -top-1/2 h-[200%] w-12 bg-white/20 transform -skew-x-12 -translate-x-full transition-transform duration-700 group-hover:translate-x-[250%]"></span>
              <span className="relative">
                {isLoading ? (
                  <span className="flex items-center justify-center">
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5 text-[#222222]"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Connecting...
                  </span>
                ) : (
                  "Login"
                )}
              </span>
            </button>
          </div>
        </form>

        {/* Register */}
        <div className="flex items-center justify-center gap-1 pt-6 border-t border-[#222222]/5">
          <span className="text-[#222222]/60 text-sm">
            Don't have an account?
          </span>
          <a
            href="/register"
            className="text-sm text-[#0B4EA2] hover:text-[#702082] font-medium transition-colors duration-200 underline-offset-2 hover:underline"
          >
            Register
          </a>
        </div>

        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-[#0B4EA2] via-[#C51E2A] to-[#F2C300] rounded-b-3xl"></div>
      </div>

      {/* Stars */}
      <div className="absolute top-20 left-1/4 opacity-20">
        <svg
          className="w-8 h-8 text-[#F2C300]"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
        </svg>
      </div>
      <div className="absolute bottom-20 right-1/4 opacity-20">
        <svg
          className="w-6 h-6 text-[#F2C300]"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
        </svg>
      </div>

      {accessModal.isVisible && (
        <WarningModal
          isVisible={accessModal.isVisible}
          onClose={handleModalClose}
          title={accessModal.title}
          message={accessModal.message}
          severity={accessModal.severity}
          buttonText="Ok"
        />
      )}
    </div>
  );
}

export default LoginPage;
