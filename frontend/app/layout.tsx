import type { Metadata } from "next";
import { Bricolage_Grotesque, JetBrains_Mono } from "next/font/google";
import NavBar from "@/components/NavBar";
import "./globals.css";

const bricolage = Bricolage_Grotesque({
  variable: "--font-bricolage",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Eidetic Memory",
  description: "Scalable long-term memory for AI agents",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${bricolage.variable} ${jetbrains.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-bg-base text-text-primary font-sans">
        <div className="dot-grid fixed inset-0 pointer-events-none z-0" aria-hidden="true" />
        <div className="relative z-10 min-h-full flex flex-col">
          <NavBar />
          <main className="flex-1">{children}</main>
        </div>
      </body>
    </html>
  );
}
