import React from "react";
import { observer } from "mobx-react-lite";
import { useConnectionStore } from "../../hooks/useStores";
import { ConnectionStatus } from "../navigation";
import type { SystemInfo } from "../../types";

export interface FooterProps {
  className?: string;
  systemInfo?: SystemInfo;
}

export const Footer = observer<FooterProps>(
  ({
    className = "",
    systemInfo = {
      version: "1.0.0",
      buildDate: new Date().toDateString(),
      environment: "development",
    },
  }) => {
    const connectionStore = useConnectionStore();

    return (
      <footer
        className={`bg-white dark:bg-secondary-800 border-t border-secondary-200 dark:border-secondary-700 ${className}`}
      >
        <div className="px-4 sm:px-6 lg:px-8 py-3">
          <div className="flex items-center justify-between text-xs text-secondary-500">
            {/* Left side - System info */}
            <div className="flex items-center space-x-4">
              <span>Billiards Trainer v{systemInfo.version}</span>
              <span className="hidden sm:inline">
                Built {new Date(systemInfo.buildDate).toLocaleDateString()}
              </span>
              {systemInfo.environment !== "production" && (
                <span className="px-2 py-1 bg-warning-100 text-warning-800 dark:bg-warning-900 dark:text-warning-200 rounded text-xs font-medium">
                  {systemInfo.environment.toUpperCase()}
                </span>
              )}
            </div>

            {/* Center - Connection status for desktop */}
            <div className="hidden md:flex">
              <ConnectionStatus showText={true} size="sm" />
            </div>
          </div>

          {/* Mobile connection status */}
          <div className="md:hidden mt-2 pt-2 border-t border-secondary-200 dark:border-secondary-700">
            <ConnectionStatus showText={true} size="sm" />
          </div>
        </div>
      </footer>
    );
  }
);
