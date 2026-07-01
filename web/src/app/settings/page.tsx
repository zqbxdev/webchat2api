"use client";

import { useEffect, useRef } from "react";
import { LoaderCircle } from "lucide-react";

import { useAuthGuard } from "@/lib/use-auth-guard";

import { BackupSettingsCard } from "./components/backup-settings-card";
import { ConfigCard } from "./components/config-card";
import { CPAPoolDialog } from "./components/cpa-pool-dialog";
import { CPAPoolsCard } from "./components/cpa-pools-card";
import { ImportBrowserDialog } from "./components/import-browser-dialog";
import { ProxyPoolCard } from "./components/proxy-pool-card";
import { SettingsHeader } from "./components/settings-header";
import { Sub2APIConnections } from "./components/sub2api-connections";
import { UserKeysCard } from "./components/user-keys-card";
import { useSettingsStore } from "./store";

function SettingsDataController() {
  const didLoadRef = useRef(false);
  const initialize = useSettingsStore((state) => state.initialize);
  const loadPools = useSettingsStore((state) => state.loadPools);
  const loadBackups = useSettingsStore((state) => state.loadBackups);
  const pools = useSettingsStore((state) => state.pools);
  const backupState = useSettingsStore((state) => state.backupState);

  useEffect(() => {
    if (didLoadRef.current) {
      return;
    }
    didLoadRef.current = true;
    void initialize();
  }, [initialize]);

  useEffect(() => {
    const hasRunningJobs = pools.some((pool) => {
      const status = pool.import_job?.status;
      return status === "pending" || status === "running";
    });
    if (!hasRunningJobs) {
      return;
    }

    const timer = window.setInterval(() => {
      void loadPools(true);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [loadPools, pools]);

  useEffect(() => {
    if (!backupState?.running) {
      return;
    }
    const timer = window.setInterval(() => {
      void loadBackups(true);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [backupState?.running, loadBackups]);

  return null;
}

function SettingsPageContent() {
  return (
    <>
      <SettingsDataController />
      <SettingsHeader />
      <section className="space-y-5 sm:space-y-6 lg:space-y-7">
        <ConfigCard />
        <ProxyPoolCard />
        <BackupSettingsCard />
        <UserKeysCard />
        <CPAPoolsCard />
        <Sub2APIConnections />
      </section>
      <CPAPoolDialog />
      <ImportBrowserDialog />
    </>
  );
}

export default function SettingsPage() {
  const { isCheckingAuth, session } = useAuthGuard(["admin"]);

  if (isCheckingAuth || !session || session.role !== "admin") {
    return (
      <div className="flex min-h-[40vh] items-center justify-center">
        <LoaderCircle className="size-5 animate-spin text-stone-400" />
      </div>
    );
  }

  return <SettingsPageContent />;
}
