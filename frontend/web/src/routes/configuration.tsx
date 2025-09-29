import { createFileRoute } from "@tanstack/react-router";
import { ConfigurationLayout, ConfigIcons } from "../components/config/ConfigurationLayout";
import { SystemSettings } from "../components/config/SystemSettings";
import { CameraConfig } from "../components/config/CameraConfig";
import { TableConfig } from "../components/config/TableConfig";
import { PhysicsConfig } from "../components/config/PhysicsConfig";
import { ProjectorConfig } from "../components/config/ProjectorConfig";
import { VisionConfig } from "../components/config/VisionConfig";
import { ConfigProfiles } from "../components/config/ConfigProfiles";
import { ConfigImportExport } from "../components/config/ConfigImportExport";
import { CalibrationWizard } from "../components/config/calibration/CalibrationWizard";

export const Route = createFileRoute("/configuration")({
  component: RouteComponent,
});

function RouteComponent() {
  const configSections = [
    {
      id: 'system',
      label: 'System Settings',
      icon: ConfigIcons.System,
      component: SystemSettings,
      description: 'Core system configuration including API, authentication, and logging settings'
    },
    {
      id: 'camera',
      label: 'Camera Configuration',
      icon: ConfigIcons.Camera,
      component: CameraConfig,
      description: 'Camera setup, resolution, image quality, and advanced camera settings'
    },
    {
      id: 'table',
      label: 'Table Configuration',
      icon: ConfigIcons.Table,
      component: TableConfig,
      description: 'Table dimensions, surface properties, pocket configuration, and environment settings'
    },
    {
      id: 'physics',
      label: 'Physics Parameters',
      icon: ConfigIcons.Physics,
      component: PhysicsConfig,
      description: 'Ball physics, collision parameters, motion simulation, and physics engine settings'
    },
    {
      id: 'projector',
      label: 'Projector Settings',
      icon: ConfigIcons.Projector,
      component: ProjectorConfig,
      description: 'Display settings, overlay configuration, positioning, and visual effects'
    },
    {
      id: 'vision',
      label: 'Vision Configuration',
      icon: ConfigIcons.Vision,
      component: VisionConfig,
      description: 'Detection thresholds, tracking parameters, prediction settings, and image processing'
    },
    {
      id: 'calibration',
      label: 'Calibration Wizard',
      icon: ConfigIcons.Calibration,
      component: CalibrationWizard,
      description: 'Step-by-step system calibration for camera, table, and projector alignment'
    },
    {
      id: 'profiles',
      label: 'Profile Management',
      icon: ConfigIcons.Profiles,
      component: ConfigProfiles,
      description: 'Manage configuration profiles, create custom profiles, and switch between settings'
    },
    {
      id: 'import-export',
      label: 'Import & Export',
      icon: ConfigIcons.ImportExport,
      component: ConfigImportExport,
      description: 'Backup and restore configuration settings, import/export profiles'
    }
  ];

  return (
    <ConfigurationLayout sections={configSections} />
  );
}
