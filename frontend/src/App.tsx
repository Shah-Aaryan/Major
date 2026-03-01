import { useEffect } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import { TradingProvider } from "@/contexts/TradingContext";

import Index from "./pages/Index";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import StrategyCreation from "./pages/StrategyCreation";
import TradingSimulation from "./pages/TradingSimulation";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

/* ---------------- Protected / Public Routes ---------------- */

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

const PublicRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated } = useAuth();

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

/* ---------------- App Routes ---------------- */

const AppRoutes = () => (
  <Routes>
    <Route path="/" element={<Index />} />

    <Route
      path="/login"
      element={
        <PublicRoute>
          <Login />
        </PublicRoute>
      }
    />

    <Route
      path="/register"
      element={
        <PublicRoute>
          <Register />
        </PublicRoute>
      }
    />

    <Route
      path="/dashboard"
      element={
        <ProtectedRoute>
          <Dashboard />
        </ProtectedRoute>
      }
    />

    <Route
      path="/create-strategy"
      element={
        <ProtectedRoute>
          <StrategyCreation />
        </ProtectedRoute>
      }
    />

    <Route
      path="/simulation"
      element={
        <ProtectedRoute>
          <TradingSimulation />
        </ProtectedRoute>
      }
    />

    <Route path="*" element={<NotFound />} />
  </Routes>
);

/* ---------------- Main App ---------------- */

const App = () => {
  // 🔒 Prevent touchpad pinch zoom & ctrl/cmd + scroll zoom
  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      // Blocks pinch zoom (trackpad) + ctrl/cmd scroll zoom
      if (e.ctrlKey || e.metaKey || Math.abs(e.deltaY) < 15) {
        e.preventDefault();
      }
    };

    const preventGesture = (e: Event) => {
      e.preventDefault();
    };

    window.addEventListener("wheel", handleWheel, { passive: false });
    window.addEventListener("gesturestart", preventGesture);
    window.addEventListener("gesturechange", preventGesture);
    window.addEventListener("gestureend", preventGesture);

    return () => {
      window.removeEventListener("wheel", handleWheel);
      window.removeEventListener("gesturestart", preventGesture);
      window.removeEventListener("gesturechange", preventGesture);
      window.removeEventListener("gestureend", preventGesture);
    };
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <AuthProvider>
          <TradingProvider>
            <Toaster />
            <Sonner />
            <BrowserRouter>
              <AppRoutes />
            </BrowserRouter>
          </TradingProvider>
        </AuthProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
