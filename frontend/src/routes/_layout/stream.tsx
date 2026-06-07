import { createFileRoute } from "@tanstack/react-router"
import {
  BrainCircuit,
  Copy,
  LocateFixed,
  Pause,
  Play,
  RadioTower,
  ShieldAlert,
  Video,
} from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import { OpenAPI } from "@/client"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export const Route = createFileRoute("/_layout/stream")({
  component: StreamPage,
  head: () => ({
    meta: [{ title: "Scout Stream" }],
  }),
})

declare global {
  interface Window {
    Hls?: {
      isSupported: () => boolean
      new (config?: Record<string, unknown>): HlsPlayer
    }
  }
}

const defaultStreamKey = "scout"
const sampleIntervalMs = 5000

type HlsPlayer = {
  loadSource: (source: string) => void
  attachMedia: (media: HTMLMediaElement) => void
  destroy: () => void
}

type SearchMatch = {
  patch_id: string
  score: number
  center_lat: number
  center_lon: number
  west: number
  south: number
  east: number
  north: number
  file_path: string
  preview_url: string | null
}

type SearchResult = {
  predicted_lat: number | null
  predicted_lon: number | null
  matches: SearchMatch[]
}

type SafetyIssue = {
  category: string
  severity: "low" | "medium" | "high" | "critical"
  confidence: number
  evidence: string
  recommended_action: string
  bounding_box: {
    x: number
    y: number
    width: number
    height: number
  }
}

type SafetyAnalysis = {
  emergency_detected: boolean
  confidence: number
  summary: string
  scene_description: string
  detected_issues: SafetyIssue[]
  limitations: string
  categories_checked: string[]
  model: string
}

function StreamPage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const analyzingRef = useRef(false)
  const inFlightRef = useRef(false)
  const understandingRef = useRef(false)
  const [streamKey, setStreamKey] = useState(defaultStreamKey)
  const [copied, setCopied] = useState<string | null>(null)
  const [playerStatus, setPlayerStatus] = useState("Waiting for stream")
  const [refreshToken] = useState(0)
  const [isAnalyzing, setIsAnalyzing] = useState(true)
  const [analysisStatus, setAnalysisStatus] = useState("Auto matching")
  const [understandingStatus, setUnderstandingStatus] = useState("Not checked")
  const [lastFrameUrl, setLastFrameUrl] = useState<string | null>(null)
  const [lastResult, setLastResult] = useState<SearchResult | null>(null)
  const [lastAnalyzedAt, setLastAnalyzedAt] = useState<string | null>(null)
  const [safetyAnalysis, setSafetyAnalysis] = useState<SafetyAnalysis | null>(
    null,
  )
  const [safetyCheckedAt, setSafetyCheckedAt] = useState<string | null>(null)

  const publicHost = window.location.host
  const rtmpUrl = useMemo(
    () =>
      `rtmp://${window.location.hostname}:1935/live/${
        streamKey || defaultStreamKey
      }`,
    [streamKey],
  )
  const hlsUrl = useMemo(
    () => `/live/${streamKey || defaultStreamKey}.m3u8`,
    [streamKey],
  )
  const absoluteHlsUrl = `${window.location.protocol}//${publicHost}${hlsUrl}`
  const bestMatch = lastResult?.matches[0] ?? null

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const media = video
    let hlsInstance: HlsPlayer | null = null
    let cancelled = false

    async function attachPlayer() {
      setPlayerStatus("Connecting")
      if (media.canPlayType("application/vnd.apple.mpegurl")) {
        media.src = hlsUrl
        setPlayerStatus("Ready")
        media.play().catch(() => setPlayerStatus("Ready - press play"))
        return
      }

      try {
        await loadHlsScript()
        if (cancelled || !window.Hls?.isSupported()) {
          setPlayerStatus("HLS unavailable in this browser")
          return
        }
        hlsInstance = new window.Hls({
          liveSyncDurationCount: 3,
          lowLatencyMode: true,
        })
        hlsInstance.loadSource(hlsUrl)
        hlsInstance.attachMedia(media)
        setPlayerStatus("Ready")
        media.play().catch(() => setPlayerStatus("Ready - press play"))
      } catch {
        setPlayerStatus("Player failed to load")
      }
    }

    attachPlayer()

    return () => {
      cancelled = true
      hlsInstance?.destroy()
      media.removeAttribute("src")
      media.load()
    }
  }, [hlsUrl, refreshToken])

  useEffect(() => {
    analyzingRef.current = isAnalyzing
  }, [isAnalyzing])

  const analyzeCurrentFrame = useCallback(async () => {
    if (inFlightRef.current) return

    const video = videoRef.current
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      setAnalysisStatus("Waiting for video")
      return
    }

    inFlightRef.current = true
    setAnalysisStatus("Embedding frame")

    try {
      const frame = await captureFrame(video)
      setLastFrameUrl((current) => {
        if (current) URL.revokeObjectURL(current)
        return URL.createObjectURL(frame)
      })
      const result = await searchFrame(frame)
      setLastResult(result)
      setLastAnalyzedAt(new Date().toLocaleTimeString())
      setAnalysisStatus(
        result.matches.length ? "Match updated" : "No indexed match",
      )
    } catch (error) {
      setAnalysisStatus(error instanceof Error ? error.message : "Search failed")
    } finally {
      inFlightRef.current = false
    }
  }, [])

  const runImageUnderstanding = useCallback(async () => {
    if (understandingRef.current) return

    const video = videoRef.current
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      setUnderstandingStatus("Waiting for video")
      return
    }

    understandingRef.current = true
    setUnderstandingStatus("Analyzing safety")

    try {
      const frame = await captureFrame(video)
      setLastFrameUrl((current) => {
        if (current) URL.revokeObjectURL(current)
        return URL.createObjectURL(frame)
      })
      const result = await analyzeSafetyFrame(frame)
      setSafetyAnalysis(result)
      setSafetyCheckedAt(new Date().toLocaleTimeString())
      setUnderstandingStatus(
        result.emergency_detected ? "Safety issue detected" : "No urgent issue",
      )
    } catch (error) {
      setUnderstandingStatus(
        error instanceof Error ? error.message : "Safety analysis failed",
      )
    } finally {
      understandingRef.current = false
    }
  }, [])

  useEffect(() => {
    if (!isAnalyzing) return

    analyzeCurrentFrame()
    const id = window.setInterval(() => {
      if (analyzingRef.current) analyzeCurrentFrame()
    }, sampleIntervalMs)

    return () => window.clearInterval(id)
  }, [analyzeCurrentFrame, isAnalyzing])

  useEffect(() => {
    return () => {
      if (lastFrameUrl) URL.revokeObjectURL(lastFrameUrl)
    }
  }, [lastFrameUrl])

  const copyValue = async (label: string, value: string) => {
    await navigator.clipboard.writeText(value)
    setCopied(label)
    window.setTimeout(() => setCopied(null), 1500)
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <RadioTower className="size-6 text-primary" />
            <h1 className="text-3xl font-semibold tracking-normal">
              Live Stream
            </h1>
          </div>
          <p className="max-w-3xl text-muted-foreground">
            Stream drone video, sample one frame every five seconds, embed it,
            and estimate position from the nearest indexed aerial imagery.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">{playerStatus}</Badge>
          <Badge variant={isAnalyzing ? "default" : "outline"}>
            {analysisStatus}
          </Badge>
        </div>
      </div>

      <Tabs defaultValue="live" className="space-y-4">
        <TabsList>
          <TabsTrigger value="live">Live</TabsTrigger>
          <TabsTrigger value="config">Config</TabsTrigger>
        </TabsList>

        <TabsContent value="live">
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.05fr)_minmax(420px,0.95fr)]">
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Drone Feed</CardTitle>
                  <CardDescription>
                    RTMP is converted to HLS for browser playback.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="relative overflow-hidden rounded-md border bg-black">
                    <video
                      className="aspect-video w-full"
                      controls
                      crossOrigin="anonymous"
                      muted
                      playsInline
                      ref={videoRef}
                    />
                    <SafetyOverlays issues={safetyAnalysis?.detected_issues ?? []} />
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button onClick={() => setIsAnalyzing((value) => !value)}>
                      {isAnalyzing ? <Pause /> : <Play />}
                      {isAnalyzing ? "Stop Matching" : "Start Matching"}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={runImageUnderstanding}
                      disabled={understandingRef.current}
                    >
                      <BrainCircuit />
                      Image Understanding
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <SafetyUnderstanding
                safetyAnalysis={safetyAnalysis}
                safetyCheckedAt={safetyCheckedAt}
                understandingStatus={understandingStatus}
              />
            </div>

            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Similarity Match</CardTitle>
                  <CardDescription>
                    Best indexed aerial patch for the latest sampled frame.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-3 md:grid-cols-2">
                    <PreviewImage
                      label="Sampled frame"
                      src={lastFrameUrl}
                      empty="No frame sampled yet."
                    />
                    <PreviewImage
                      label="Best match"
                      src={bestMatch?.preview_url ?? null}
                      empty="No match yet."
                      authenticated
                    />
                  </div>
                  {bestMatch ? (
                    <div className="grid gap-3 rounded-md border p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="text-sm text-muted-foreground">
                            Estimated position
                          </p>
                          <p className="text-xl font-semibold tabular-nums">
                            {bestMatch.center_lat.toFixed(6)},{" "}
                            {bestMatch.center_lon.toFixed(6)}
                          </p>
                        </div>
                        <Badge variant="outline">
                          score {bestMatch.score.toFixed(5)}
                        </Badge>
                      </div>
                      {lastAnalyzedAt ? (
                        <p className="text-xs text-muted-foreground">
                          Last analyzed at {lastAnalyzedAt}
                        </p>
                      ) : null}
                    </div>
                  ) : (
                    <div className="rounded-md border border-dashed p-6 text-sm text-muted-foreground">
                      Matching starts automatically once the stream is visible.
                      Results depend on the aerial image database being indexed
                      first.
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Estimated Location</CardTitle>
                  <CardDescription>
                    Map centers on the highest-similarity patch.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {bestMatch ? (
                    <iframe
                      className="aspect-[4/3] w-full rounded-md border"
                      loading="lazy"
                      referrerPolicy="no-referrer-when-downgrade"
                      src={openStreetMapEmbedUrl(bestMatch)}
                      title="Estimated drone location map"
                    />
                  ) : (
                    <div className="flex aspect-[4/3] items-center justify-center rounded-md border border-dashed text-sm text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <LocateFixed className="size-4" />
                        Waiting for a geolocation estimate
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="config">
          <Card>
            <CardHeader>
              <CardTitle>RTMP Publisher</CardTitle>
              <CardDescription>
                Use this endpoint from DJI Fly, OBS, or ffmpeg.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 lg:grid-cols-2">
              <div className="grid gap-2 lg:col-span-2">
                <Label htmlFor="stream-key">Stream key</Label>
                <Input
                  id="stream-key"
                  value={streamKey}
                  onChange={(event) =>
                    setStreamKey(event.target.value.trim() || defaultStreamKey)
                  }
                />
              </div>
              <Endpoint
                icon={RadioTower}
                label="RTMP URL"
                value={rtmpUrl}
                copied={copied === "rtmp"}
                onCopy={() => copyValue("rtmp", rtmpUrl)}
              />
              <Endpoint
                icon={Video}
                label="Playback URL"
                value={absoluteHlsUrl}
                copied={copied === "playback"}
                onCopy={() => copyValue("playback", absoluteHlsUrl)}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

function SafetyOverlays({ issues }: { issues: SafetyIssue[] }) {
  if (!issues.length) return null

  return (
    <div className="pointer-events-none absolute inset-0">
      {issues.map((issue) => {
        const box = clampBox(issue.bounding_box)
        return (
          <div
            className={`absolute border-2 ${severityBorder(issue.severity)}`}
            key={`${issue.category}-${issue.evidence}`}
            style={{
              left: `${box.x * 100}%`,
              top: `${box.y * 100}%`,
              width: `${box.width * 100}%`,
              height: `${box.height * 100}%`,
            }}
          >
            <div className="absolute -top-7 left-0 max-w-full truncate rounded-sm bg-black/80 px-2 py-1 text-xs font-medium text-white">
              {issue.category}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function SafetyUnderstanding({
  safetyAnalysis,
  safetyCheckedAt,
  understandingStatus,
}: {
  safetyAnalysis: SafetyAnalysis | null
  safetyCheckedAt: string | null
  understandingStatus: string
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Safety Understanding</CardTitle>
        <CardDescription>
          Single-frame OpenAI vision check, triggered manually.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <Badge
            variant={safetyAnalysis?.emergency_detected ? "destructive" : "outline"}
          >
            {understandingStatus}
          </Badge>
          {safetyAnalysis ? (
            <Badge variant="outline">
              confidence {Math.round(safetyAnalysis.confidence * 100)}%
            </Badge>
          ) : null}
          {safetyCheckedAt ? (
            <span className="text-xs text-muted-foreground">
              Checked at {safetyCheckedAt}
            </span>
          ) : null}
        </div>

        {safetyAnalysis ? (
          <div className="space-y-4">
            <div className="rounded-md border p-4">
              <div className="mb-2 flex items-center gap-2">
                <ShieldAlert className="size-4 text-muted-foreground" />
                <p className="font-medium">
                  Emergency:{" "}
                  {safetyAnalysis.emergency_detected ? "true" : "false"}
                </p>
              </div>
              <p className="text-sm">{safetyAnalysis.summary}</p>
              <p className="mt-2 text-xs text-muted-foreground">
                {safetyAnalysis.scene_description}
              </p>
            </div>

            <div className="space-y-2">
              <Label>Detected issues</Label>
              {safetyAnalysis.detected_issues.length ? (
                <div className="space-y-2">
                  {safetyAnalysis.detected_issues.map((issue) => (
                    <div
                      className="rounded-md border p-3 text-sm"
                      key={`${issue.category}-${issue.evidence}`}
                    >
                      <div className="mb-1 flex flex-wrap items-center gap-2">
                        <Badge variant="outline">{issue.severity}</Badge>
                        <p className="font-medium">{issue.category}</p>
                        <span className="text-xs text-muted-foreground">
                          {Math.round(issue.confidence * 100)}%
                        </span>
                      </div>
                      <p>{issue.evidence}</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {issue.recommended_action}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
                  No visible urgent safety issue detected in this frame.
                </div>
              )}
            </div>

            <div className="space-y-2">
              <Label>Monitored categories</Label>
              <div className="flex flex-wrap gap-2">
                {safetyAnalysis.categories_checked.map((category) => (
                  <Badge variant="secondary" key={category}>
                    {category}
                  </Badge>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                {safetyAnalysis.limitations}
              </p>
            </div>
          </div>
        ) : (
          <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
            Click Image Understanding to analyze the current frame for safety
            issues.
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function Endpoint({
  copied,
  icon: Icon,
  label,
  onCopy,
  value,
}: {
  copied: boolean
  icon: typeof RadioTower
  label: string
  onCopy: () => void
  value: string
}) {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <div className="flex gap-2">
        <div className="flex min-w-0 flex-1 items-center gap-2 rounded-md border bg-muted px-3 py-2">
          <Icon className="size-4 shrink-0 text-muted-foreground" />
          <code className="truncate text-xs">{value}</code>
        </div>
        <Button aria-label={`Copy ${label}`} size="icon" onClick={onCopy}>
          {copied ? <span className="text-xs">OK</span> : <Copy />}
        </Button>
      </div>
    </div>
  )
}

function PreviewImage({
  authenticated = false,
  empty,
  label,
  src,
}: {
  authenticated?: boolean
  empty: string
  label: string
  src: string | null
}) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!authenticated || !src) {
      setObjectUrl(null)
      return
    }

    let active = true
    let createdUrl: string | null = null
    scoutBlob(src).then((blob) => {
      if (!active) return
      createdUrl = URL.createObjectURL(blob)
      setObjectUrl(createdUrl)
    })

    return () => {
      active = false
      if (createdUrl) URL.revokeObjectURL(createdUrl)
    }
  }, [authenticated, src])

  const displaySrc = authenticated ? objectUrl : src

  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      {displaySrc ? (
        <img
          alt={label}
          className="aspect-video w-full rounded-md border object-cover"
          src={displaySrc}
        />
      ) : (
        <div className="flex aspect-video items-center justify-center rounded-md border border-dashed bg-muted/40 p-4 text-center text-sm text-muted-foreground">
          {empty}
        </div>
      )}
    </div>
  )
}

async function captureFrame(video: HTMLVideoElement): Promise<Blob> {
  const width = video.videoWidth
  const height = video.videoHeight
  if (!width || !height) throw new Error("Video dimensions unavailable")

  const canvas = document.createElement("canvas")
  const maxWidth = 960
  const scale = Math.min(1, maxWidth / width)
  canvas.width = Math.round(width * scale)
  canvas.height = Math.round(height * scale)
  const context = canvas.getContext("2d")
  if (!context) throw new Error("Canvas unavailable")
  context.drawImage(video, 0, 0, canvas.width, canvas.height)

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Frame capture failed"))
          return
        }
        resolve(blob)
      },
      "image/jpeg",
      0.88,
    )
  })
}

async function searchFrame(frame: Blob): Promise<SearchResult> {
  const payload = new FormData()
  payload.append("image", frame, `stream-frame-${Date.now()}.jpg`)
  payload.append("top_k", "5")
  return scoutJson<SearchResult>("/api/v1/scout/search/image", {
    method: "POST",
    body: payload,
  })
}

async function analyzeSafetyFrame(frame: Blob): Promise<SafetyAnalysis> {
  const payload = new FormData()
  payload.append("image", frame, `safety-frame-${Date.now()}.jpg`)
  return scoutJson<SafetyAnalysis>("/api/v1/scout/analyze/safety", {
    method: "POST",
    body: payload,
  })
}

async function scoutJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${OpenAPI.BASE}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access_token") ?? ""}`,
      ...(init?.headers ?? {}),
    },
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.detail ?? `Scout request failed: ${response.status}`)
  }
  return response.json()
}

async function scoutBlob(path: string): Promise<Blob> {
  const response = await fetch(`${OpenAPI.BASE}${path}`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access_token") ?? ""}`,
    },
  })
  if (!response.ok) {
    throw new Error(`Scout image request failed: ${response.status}`)
  }
  return response.blob()
}

function openStreetMapEmbedUrl(match: SearchMatch) {
  const span = 0.006
  const west = match.center_lon - span
  const east = match.center_lon + span
  const south = match.center_lat - span
  const north = match.center_lat + span
  const params = new URLSearchParams({
    bbox: `${west},${south},${east},${north}`,
    layer: "mapnik",
    marker: `${match.center_lat},${match.center_lon}`,
  })
  return `https://www.openstreetmap.org/export/embed.html?${params.toString()}`
}

function loadHlsScript() {
  if (window.Hls) return Promise.resolve()

  return new Promise<void>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      'script[data-scout-hls="true"]',
    )
    if (existing) {
      existing.addEventListener("load", () => resolve(), { once: true })
      existing.addEventListener("error", () => reject(), { once: true })
      return
    }

    const script = document.createElement("script")
    script.src = "https://cdn.jsdelivr.net/npm/hls.js@1.6.15/dist/hls.min.js"
    script.async = true
    script.dataset.scoutHls = "true"
    script.onload = () => resolve()
    script.onerror = () => reject()
    document.head.appendChild(script)
  })
}

function clampBox(box: SafetyIssue["bounding_box"]) {
  const x = clamp01(box.x)
  const y = clamp01(box.y)
  const width = Math.min(clamp01(box.width), 1 - x)
  const height = Math.min(clamp01(box.height), 1 - y)
  return { x, y, width, height }
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value))
}

function severityBorder(severity: SafetyIssue["severity"]) {
  if (severity === "critical") return "border-red-500"
  if (severity === "high") return "border-orange-500"
  if (severity === "medium") return "border-yellow-400"
  return "border-sky-400"
}
