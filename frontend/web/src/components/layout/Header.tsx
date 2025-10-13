import React, { useState } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import { observer } from "mobx-react-lite";
import { useUIStore } from "../../hooks/useStores";
import { ConnectionStatus, NotificationCenter } from "../navigation";
import { Button } from "../ui/Button";
import type { NavItem } from "../../types";

export interface HeaderProps {
  className?: string;
}

// Navigation items
const navigationItems: NavItem[] = [
  {
    id: "home",
    label: "Dashboard",
    icon: "home",
    path: "/",
  },
  {
    id: "calibration",
    label: "Calibration",
    icon: "calibration",
    path: "/calibration",
  },
  {
    id: "configuration",
    label: "Configuration",
    icon: "configuration",
    path: "/configuration",
  },
  {
    id: "diagnostics",
    label: "Diagnostics",
    icon: "diagnostics",
    path: "/diagnostics",
  },
];

const NavIcon: React.FC<{ name?: string; className?: string }> = ({
  name,
  className = "w-5 h-5",
}) => {
  if (!name) return null;

  const icons: Record<string, React.ReactNode> = {
    home: (
      <svg
        className={className}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
        />
      </svg>
    ),
    calibration: (
      <svg
        className={className}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
        />
      </svg>
    ),
    configuration: (
      <svg
        className={className}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
        />
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
        />
      </svg>
    ),
    diagnostics: (
      <svg
        className={className}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
        />
      </svg>
    ),
  };

  return icons[name] || null;
};

export const Header = observer<HeaderProps>(({ className = "" }) => {
  const uiStore = useUIStore();
  const router = useRouterState();
  const currentPath = router.location.pathname;
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header
      className={`bg-white dark:bg-secondary-800 border-b border-secondary-200 dark:border-secondary-700 ${className}`}
    >
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left side - Logo and title */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <img src="/logo192.png" alt="Logo" className="w-6 h-6" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-lg font-semibold text-secondary-900 dark:text-secondary-100">
                  Billiards Trainer
                </h1>
              </div>
            </div>
          </div>

          {/* Center - Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navigationItems.map((item) => {
              const isActive = currentPath === item.path;
              return (
                <Link
                  key={item.id}
                  to={item.path}
                  className={`flex items-center px-4 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
                    isActive
                      ? "bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100"
                      : "text-secondary-700 hover:bg-secondary-100 hover:text-secondary-900 dark:text-secondary-300 dark:hover:bg-secondary-700 dark:hover:text-secondary-100"
                  } ${item.disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                  onClick={
                    item.disabled ? (e) => e.preventDefault() : undefined
                  }
                  aria-disabled={item.disabled}
                  aria-current={isActive ? "page" : undefined}
                >
                  <NavIcon name={item.icon} className="w-5 h-5 mr-2" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* Right side */}
          <div className="flex items-center space-x-2">
            {/* Connection Status */}
            <ConnectionStatus
              className="hidden sm:flex"
              showText={false}
              size="md"
            />

            {/* Notifications */}
            <NotificationCenter />

            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden"
              aria-label="Toggle mobile menu"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                {mobileMenuOpen ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                ) : (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                )}
              </svg>
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden py-3 space-y-1 border-t border-secondary-200 dark:border-secondary-700">
            {navigationItems.map((item) => {
              const isActive = currentPath === item.path;
              return (
                <Link
                  key={item.id}
                  to={item.path}
                  className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
                    isActive
                      ? "bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100"
                      : "text-secondary-700 hover:bg-secondary-100 hover:text-secondary-900 dark:text-secondary-300 dark:hover:bg-secondary-700 dark:hover:text-secondary-100"
                  } ${item.disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                  onClick={(e) => {
                    if (item.disabled) {
                      e.preventDefault();
                    } else {
                      setMobileMenuOpen(false);
                    }
                  }}
                  aria-disabled={item.disabled}
                  aria-current={isActive ? "page" : undefined}
                >
                  <NavIcon name={item.icon} className="w-5 h-5 mr-3" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>
        )}
      </div>

      {/* Mobile connection status */}
      <div className="sm:hidden px-4 pb-2 border-t border-secondary-200 dark:border-secondary-700">
        <ConnectionStatus showText={true} size="sm" />
      </div>
    </header>
  );
});
