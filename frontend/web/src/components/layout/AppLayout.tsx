import { observer } from "mobx-react-lite";
import React, { useId } from "react";
import { useKeyboardDetection } from "../../hooks/useAccessibility";
import { useUIStore } from "../../hooks/useStores";
import { AnnouncementRegion } from "../accessibility/AnnouncementRegion";
import { SkipLinks, SkipLinkTarget } from "../accessibility/SkipLinks";
import { FullScreenLoading } from "../ui/LoadingSpinner";
import { Header } from "./Header";
import { MainContent } from "./MainContent";

export interface AppLayoutProps {
  children: React.ReactNode;
  className?: string;
}

export const AppLayout = observer<AppLayoutProps>(
  ({ children, className = "" }) => {
    const uiStore = useUIStore();
    const isKeyboardUser = useKeyboardDetection();

    return (
      <div
        className={`min-h-screen bg-gray-50 dark:bg-gray-900 ${className} ${isKeyboardUser ? "keyboard-user" : ""}`}
      >
        {/* Skip Links for keyboard navigation */}
        <SkipLinks />

        {/* Accessibility Announcement Region */}
        <AnnouncementRegion />

        {/* Global Loading Overlay */}
        {uiStore.isGlobalLoading && (
          <FullScreenLoading text={uiStore.loadingText} />
        )}

        {/* Header */}
        <SkipLinkTarget id={useId()} as="header">
          <Header />
        </SkipLinkTarget>

        {/* Main Content Area */}
        <div className="flex flex-col min-h-screen">
          {/* Main content */}
          <SkipLinkTarget id={useId()} as="main" className="flex-1">
            <MainContent>{children}</MainContent>
          </SkipLinkTarget>
        </div>
      </div>
    );
  }
);
