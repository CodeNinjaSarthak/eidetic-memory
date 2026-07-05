"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/demo", label: "Demo" },
] as const;

export default function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="bg-bg-surface/80 backdrop-blur-sm border-b border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          <Link href="/" className="text-lg font-bold text-text-primary tracking-tight">
            Eidetic Memory
          </Link>
          <div className="flex gap-4">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`relative px-3 py-2 text-sm font-medium transition-colors ${
                  pathname.startsWith(link.href)
                    ? "text-accent-teal"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                {link.label}
                {pathname.startsWith(link.href) && (
                  <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent-teal origin-left animate-[slideIn_0.2s_ease-out]" />
                )}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
