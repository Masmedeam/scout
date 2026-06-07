import { Link } from "@tanstack/react-router"
import { Radar } from "lucide-react"

import { cn } from "@/lib/utils"

interface LogoProps {
  variant?: "full" | "icon" | "responsive"
  className?: string
  asLink?: boolean
}

export function Logo({
  variant = "full",
  className,
  asLink = true,
}: LogoProps) {
  const content =
    variant === "responsive" ? (
      <>
        <span
          className={cn(
            "flex items-center gap-2 text-lg font-semibold group-data-[collapsible=icon]:hidden",
            className,
          )}
        >
          <Radar className="size-5 text-primary" />
          Scout
        </span>
        <Radar
          className={cn(
            "hidden size-5 text-primary group-data-[collapsible=icon]:block",
            className,
          )}
        />
      </>
    ) : (
      <span
        className={cn(
          "flex items-center gap-2 font-semibold",
          variant === "icon" && "gap-0",
          className,
        )}
      >
        <Radar className="size-5 text-primary" />
        {variant === "full" ? "Scout" : null}
      </span>
    )

  if (!asLink) {
    return content
  }

  return <Link to="/">{content}</Link>
}
