import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  email: string;
  token: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  register: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    const storedToken = localStorage.getItem('auth_token');
    const storedEmail = localStorage.getItem('user_email');
    if (storedToken && storedEmail) {
      setUser({ email: storedEmail, token: storedToken });
    }
  }, []);

  const login = async (email: string, password: string): Promise<boolean> => {
    // Mock authentication - accept any valid email format
    if (email && password.length >= 6) {
      const mockToken = `mock_token_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('auth_token', mockToken);
      localStorage.setItem('user_email', email);
      setUser({ email, token: mockToken });
      return true;
    }
    return false;
  };

  const register = async (email: string, password: string): Promise<boolean> => {
    // Mock registration
    if (email && password.length >= 6) {
      const mockToken = `mock_token_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('auth_token', mockToken);
      localStorage.setItem('user_email', email);
      setUser({ email, token: mockToken });
      return true;
    }
    return false;
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_email');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: !!user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
