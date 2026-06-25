"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState } from "react"
import type { LucideIcon } from "lucide-react"
import { Activity, BookOpen, Home, Layers, Menu, ShieldCheck, X } from "lucide-react"

import { cn } from "@/lib/utils"

type NavItem = {
  href: string
  label: string
  description: string
  icon: LucideIcon
}

const NAV_ITEMS: NavItem[] = [
  {
    href: "/",
    label: "Covert overhead",
    description: "Cluster simulator",
    icon: Home,
  },
  {
    href: "/leakage",
    label: "Endpoint leakage",
    description: "Clean vs dirty",
    icon: Activity,
  },
  {
    href: "/leakage/bound",
    label: "Capacity bound",
    description: "Compartment bound",
    icon: ShieldCheck,
  },
  {
    href: "/audit-batching",
    label: "Replay audit",
    description: "Physical batching",
    icon: Layers,
  },
  {
    href: "/proofs",
    label: "Proof cost",
    description: "Appendix figure",
    icon: BookOpen,
  },
]

function isActivePath(pathname: string, href: string) {
  return pathname === href
}

function SidebarNav({
  pathname,
  onNavigate,
}: {
  pathname: string
  onNavigate?: () => void
}) {
  return (
    <nav className="space-y-1 px-3" aria-label="Primary navigation">
      {NAV_ITEMS.map(item => {
        const Icon = item.icon
        const active = isActivePath(pathname, item.href)

        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            className={cn(
              "group flex min-h-12 items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
              active
                ? "bg-sidebar-accent text-sidebar-accent-foreground"
                : "text-sidebar-foreground/72 hover:bg-sidebar-accent/70 hover:text-sidebar-accent-foreground"
            )}
          >
            <Icon className="h-4 w-4 shrink-0" />
            <span className="min-w-0">
              <span className="block truncate font-medium leading-5">{item.label}</span>
              <span className="block truncate text-xs leading-4 text-muted-foreground">
                {item.description}
              </span>
            </span>
          </Link>
        )
      })}
    </nav>
  )
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const [mobileOpen, setMobileOpen] = useState(false)

  if (pathname.startsWith("/embed")) {
    return <>{children}</>
  }

  return (
    <div className="min-h-screen bg-background">
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-64 border-r border-sidebar-border bg-sidebar text-sidebar-foreground lg:flex lg:flex-col">
        <div className="border-b border-sidebar-border px-5 py-5">
          <Link href="/" className="block">
            <div className="text-sm font-semibold tracking-tight">Gwangju</div>
            <div className="mt-1 text-xs text-muted-foreground">Cluster simulation tools</div>
          </Link>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto py-3">
          <SidebarNav pathname={pathname} />
        </div>
      </aside>

      <div className="min-w-0 lg:pl-64">
        <header className="sticky top-0 z-40 flex h-14 items-center justify-between border-b border-border bg-background/95 px-4 backdrop-blur lg:hidden">
          <Link href="/" className="font-semibold tracking-tight">
            Gwangju
          </Link>
          <button
            type="button"
            onClick={() => setMobileOpen(true)}
            className="inline-flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label="Open navigation"
          >
            <Menu className="h-5 w-5" />
          </button>
        </header>

        <main>{children}</main>
      </div>

      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            className="absolute inset-0 bg-background/80 backdrop-blur-sm"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
          />
          <div className="absolute inset-y-0 left-0 flex w-[min(20rem,calc(100vw-3rem))] flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground shadow-xl">
            <div className="flex h-14 items-center justify-between border-b border-sidebar-border px-4">
              <div>
                <div className="text-sm font-semibold tracking-tight">Gwangju</div>
                <div className="text-xs text-muted-foreground">Cluster simulation tools</div>
              </div>
              <button
                type="button"
                onClick={() => setMobileOpen(false)}
                className="inline-flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                aria-label="Close navigation"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto py-3">
              <SidebarNav pathname={pathname} onNavigate={() => setMobileOpen(false)} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
