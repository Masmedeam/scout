import { createFileRoute } from "@tanstack/react-router"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Activity,
  Database,
  ImagePlus,
  MapPin,
  Radar,
  Search,
  Upload,
  Video,
} from "lucide-react"
import { type FormEvent, useEffect, useState } from "react"

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

export const Route = createFileRoute("/_layout/")({
  component: Dashboard,
  head: () => ({
    meta: [
      {
        title: "Scout",
      },
    ],
  }),
})

type ScoutStatus = {
  locations: number
  raster_assets: number
  patches: number
  redis_available: boolean
  redis_index: string
  embedding_model: string
  embedding_dim: number
}

type ImportResult = {
  patches_created: number
  redis_indexed: number
  location: { name: string }
  raster_asset: { width: number; height: number }
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
  retrieval_score?: number | null
  fine_match_score?: number | null
  fine_match_inliers?: number | null
  refined_lat?: number | null
  refined_lon?: number | null
}

type SearchResult = {
  predicted_lat: number | null
  predicted_lon: number | null
  confidence?: number | null
  method?: string | null
  matches: SearchMatch[]
}

type ExampleImage = {
  id: string
  label: string
  center_lat: number
  center_lon: number
  preview_url: string
}

type ScoutCoverage = {
  coverage_percent: number
  patch_count: number
  raster_asset_count: number
  uncovered_cell_count: number
}

type LiveTrackPoint = {
  frame_id: string
  lat: number
  lon: number
  source: string
  vps_confidence: number
  timestamp_s: number
}

type LiveSessionResult = {
  session_id: string
  status: string
  frame_count: number
  vps_fix_count: number
  median_confidence: number
  track: LiveTrackPoint[]
  error?: string | null
}

function Dashboard() {
  const queryClient = useQueryClient()
  const [importFile, setImportFile] = useState<File | null>(null)
  const [queryFile, setQueryFile] = useState<File | null>(null)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [altitudeM, setAltitudeM] = useState("120")
  const [importResult, setImportResult] = useState<ImportResult | null>(null)
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null)
  const [liveResult, setLiveResult] = useState<LiveSessionResult | null>(null)
  const [form, setForm] = useState({
    location_name: "San Francisco",
    west: "-122.515",
    south: "37.703",
    east: "-122.355",
    north: "37.812",
    patch_size: "256",
    overlap: "64",
    source_provider: "local",
    license: "open imagery",
  })

  const statusQuery = useQuery({
    queryKey: ["scout-status"],
    queryFn: () => scoutJson<ScoutStatus>("/api/v1/scout/index/status"),
  })

  const coverageQuery = useQuery({
    queryKey: ["scout-coverage"],
    queryFn: () => scoutJson<ScoutCoverage>("/api/v1/scout/index/coverage"),
  })

  const examplesQuery = useQuery({
    queryKey: ["scout-examples"],
    queryFn: () => scoutJson<ExampleImage[]>("/api/v1/scout/examples?limit=8"),
  })

  const importMutation = useMutation({
    mutationFn: async () => {
      if (!importFile) {
        throw new Error("Choose an imagery file first.")
      }
      const payload = new FormData()
      payload.append("image", importFile)
      for (const [key, value] of Object.entries(form)) {
        payload.append(key, value)
      }
      return scoutJson<ImportResult>("/api/v1/scout/imagery/import", {
        method: "POST",
        body: payload,
      })
    },
    onSuccess: (data) => {
      setImportResult(data)
      queryClient.invalidateQueries({ queryKey: ["scout-status"] })
    },
  })

  const searchMutation = useMutation({
    mutationFn: async (input?: { file: File; filename?: string }) => {
      const file = input?.file ?? queryFile
      if (!file) {
        throw new Error("Choose a query image first.")
      }
      const payload = new FormData()
      payload.append("image", file, input?.filename ?? file.name)
      payload.append("top_k", "5")
      return scoutJson<SearchResult>("/api/v1/scout/search/image", {
        method: "POST",
        body: payload,
      })
    },
    onSuccess: setSearchResult,
  })

  const liveMutation = useMutation({
    mutationFn: async () => {
      if (!videoFile) {
        throw new Error("Choose a drone video first.")
      }
      const payload = new FormData()
      payload.append("video", videoFile)
      payload.append("altitude_m", altitudeM)
      payload.append("fps", "2")
      return scoutJson<LiveSessionResult>("/api/v1/scout/live/session", {
        method: "POST",
        body: payload,
      })
    },
    onSuccess: setLiveResult,
  })

  const handleImport = (event: FormEvent) => {
    event.preventDefault()
    importMutation.mutate()
  }

  const handleSearch = (event: FormEvent) => {
    event.preventDefault()
    searchMutation.mutate(undefined)
  }

  const handleLive = (event: FormEvent) => {
    event.preventDefault()
    liveMutation.mutate()
  }

  const downloadGeoJson = () => {
    if (!liveResult?.track.length) {
      return
    }
    const featureCollection = {
      type: "FeatureCollection",
      features: liveResult.track.map((point) => ({
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [point.lon, point.lat],
        },
        properties: {
          frame_id: point.frame_id,
          source: point.source,
          vps_confidence: point.vps_confidence,
          timestamp_s: point.timestamp_s,
        },
      })),
    }
    const blob = new Blob([JSON.stringify(featureCollection, null, 2)], {
      type: "application/geo+json",
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = `scout-track-${liveResult.session_id}.geojson`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const handleExampleSearch = async (example: ExampleImage) => {
    const blob = await scoutBlob(example.preview_url)
    const file = new File([blob], `${example.label}.jpg`, { type: blob.type })
    searchMutation.mutate({ file, filename: `${example.id}.jpg` })
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Radar className="size-6 text-primary" />
            <h1 className="text-3xl font-semibold tracking-normal">Scout</h1>
          </div>
          <p className="max-w-3xl text-muted-foreground">
            Build a georeferenced image embedding index for drone visual
            positioning. This MVP imports aerial imagery, tiles it, indexes
            patch embeddings in Redis, and searches with a test image.
          </p>
        </div>
        <Badge variant={statusQuery.data?.redis_available ? "default" : "outline"}>
          Redis {statusQuery.data?.redis_available ? "online" : "unavailable"}
        </Badge>
      </div>

      <div className="grid gap-4 md:grid-cols-5">
        <MetricCard
          icon={MapPin}
          label="Locations"
          value={statusQuery.data?.locations ?? 0}
        />
        <MetricCard
          icon={ImagePlus}
          label="Raster Assets"
          value={statusQuery.data?.raster_assets ?? 0}
        />
        <MetricCard
          icon={Database}
          label="Indexed Patches"
          value={statusQuery.data?.patches ?? 0}
        />
        <MetricCard
          icon={Activity}
          label="Embedding Dim"
          value={statusQuery.data?.embedding_dim ?? 256}
        />
        <MetricCard
          icon={Radar}
          label="SF Coverage"
          value={`${(coverageQuery.data?.coverage_percent ?? 0).toFixed(0)}%`}
        />
      </div>

      <Tabs defaultValue="dataset" className="space-y-4">
        <TabsList>
          <TabsTrigger value="dataset">
            <Database />
            Dataset
          </TabsTrigger>
          <TabsTrigger value="search">
            <Search />
            Search
          </TabsTrigger>
          <TabsTrigger value="live">
            <Video />
            Live
          </TabsTrigger>
        </TabsList>

        <TabsContent value="dataset">
          <Card>
            <CardHeader>
              <CardTitle>Import Georeferenced Imagery</CardTitle>
              <CardDescription>
                Upload an aerial image and its geographic bounds. Scout will
                create overlapping tiles and push embeddings into Redis.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form className="grid gap-4" onSubmit={handleImport}>
                <div className="grid gap-4 md:grid-cols-2">
                  <Field
                    label="Location"
                    value={form.location_name}
                    onChange={(value) =>
                      setForm((current) => ({
                        ...current,
                        location_name: value,
                      }))
                    }
                  />
                  <Field
                    label="Source"
                    value={form.source_provider}
                    onChange={(value) =>
                      setForm((current) => ({
                        ...current,
                        source_provider: value,
                      }))
                    }
                  />
                  <Field
                    label="West"
                    value={form.west}
                    onChange={(value) =>
                      setForm((current) => ({ ...current, west: value }))
                    }
                  />
                  <Field
                    label="South"
                    value={form.south}
                    onChange={(value) =>
                      setForm((current) => ({ ...current, south: value }))
                    }
                  />
                  <Field
                    label="East"
                    value={form.east}
                    onChange={(value) =>
                      setForm((current) => ({ ...current, east: value }))
                    }
                  />
                  <Field
                    label="North"
                    value={form.north}
                    onChange={(value) =>
                      setForm((current) => ({ ...current, north: value }))
                    }
                  />
                  <Field
                    label="Patch Size"
                    value={form.patch_size}
                    onChange={(value) =>
                      setForm((current) => ({ ...current, patch_size: value }))
                    }
                  />
                  <Field
                    label="Overlap"
                    value={form.overlap}
                    onChange={(value) =>
                      setForm((current) => ({ ...current, overlap: value }))
                    }
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="imagery">Imagery file</Label>
                  <Input
                    id="imagery"
                    type="file"
                    accept="image/*"
                    onChange={(event) =>
                      setImportFile(event.target.files?.[0] ?? null)
                    }
                  />
                </div>
                <Button
                  className="w-fit"
                  type="submit"
                  disabled={importMutation.isPending}
                >
                  <Upload />
                  {importMutation.isPending ? "Indexing..." : "Import and Index"}
                </Button>
                {importMutation.error ? (
                  <p className="text-sm text-destructive">
                    {importMutation.error.message}
                  </p>
                ) : null}
                {importResult ? (
                  <p className="text-sm text-muted-foreground">
                    Imported {importResult.raster_asset.width} x{" "}
                    {importResult.raster_asset.height} imagery for{" "}
                    {importResult.location.name}. Created{" "}
                    {importResult.patches_created} patches and indexed{" "}
                    {importResult.redis_indexed} in Redis.
                  </p>
                ) : null}
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="search">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,420px)_1fr]">
            <Card>
              <CardHeader>
                <CardTitle>Query Image</CardTitle>
                <CardDescription>
                  Upload a cropped aerial/drone-like image to retrieve nearest
                  geospatial patches.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form className="grid gap-4" onSubmit={handleSearch}>
                  <div className="grid gap-2">
                    <Label htmlFor="query-image">Test image</Label>
                    <Input
                      id="query-image"
                      type="file"
                      accept="image/*"
                      onChange={(event) =>
                        setQueryFile(event.target.files?.[0] ?? null)
                      }
                    />
                  </div>
                  <Button type="submit" disabled={searchMutation.isPending}>
                    <Search />
                    {searchMutation.isPending ? "Searching..." : "Search Index"}
                  </Button>
                  {searchMutation.error ? (
                    <p className="text-sm text-destructive">
                      {searchMutation.error.message}
                    </p>
                  ) : null}
                </form>
                <div className="mt-6 space-y-3">
                  <div>
                    <h3 className="text-sm font-medium">Seeded Examples</h3>
                    <p className="text-xs text-muted-foreground">
                      Holdout SF crops with different zoom and light transforms.
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {examplesQuery.data?.map((example) => (
                      <button
                        className="group overflow-hidden rounded-md border bg-background text-left transition-colors hover:bg-accent"
                        disabled={searchMutation.isPending}
                        key={example.id}
                        onClick={() => handleExampleSearch(example)}
                        type="button"
                      >
                        <AuthImage
                          alt={example.label}
                          className="aspect-square w-full object-cover"
                          src={example.preview_url}
                        />
                        <div className="space-y-1 p-2">
                          <p className="truncate text-xs font-medium">
                            {example.label}
                          </p>
                          <p className="truncate text-[11px] text-muted-foreground">
                            {example.center_lat.toFixed(4)},{" "}
                            {example.center_lon.toFixed(4)}
                          </p>
                        </div>
                      </button>
                    ))}
                  </div>
                  {examplesQuery.data?.length === 0 ? (
                    <p className="text-xs text-muted-foreground">
                      Seed SF data to populate examples.
                    </p>
                  ) : null}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Match Results</CardTitle>
                <CardDescription>
                  Lower score is closer for the current Redis cosine index.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {searchResult?.matches.length ? (
                  <div className="space-y-4">
                    <div className="rounded-md border p-4">
                      <p className="text-sm text-muted-foreground">
                        Estimated position
                        {searchResult.method ? ` (${searchResult.method})` : ""}
                      </p>
                      <p className="text-xl font-semibold">
                        {searchResult.predicted_lat?.toFixed(6)},{" "}
                        {searchResult.predicted_lon?.toFixed(6)}
                      </p>
                      {searchResult.confidence != null ? (
                        <p className="text-sm text-muted-foreground">
                          Confidence {(searchResult.confidence * 100).toFixed(0)}%
                        </p>
                      ) : null}
                      {searchResult.matches[0]?.preview_url ? (
                        <AuthImage
                          alt="Best matched location preview"
                          className="mt-4 aspect-video w-full rounded-md border object-cover"
                          src={searchResult.matches[0].preview_url}
                        />
                      ) : null}
                    </div>
                    <div className="grid gap-3">
                      {searchResult.matches.map((match, index) => (
                        <div
                          className="grid gap-3 rounded-md border p-3 md:grid-cols-[112px_auto_1fr_auto]"
                          key={match.patch_id}
                        >
                          {match.preview_url ? (
                            <AuthImage
                              alt={`Match ${index + 1} preview`}
                              className="aspect-square w-full rounded-md border object-cover"
                              src={match.preview_url}
                            />
                          ) : (
                            <div className="aspect-square rounded-md border bg-muted" />
                          )}
                          <Badge variant={index === 0 ? "default" : "outline"}>
                            #{index + 1}
                          </Badge>
                          <div>
                            <p className="font-medium">
                              {(match.refined_lat ?? match.center_lat).toFixed(6)},{" "}
                              {(match.refined_lon ?? match.center_lon).toFixed(6)}
                            </p>
                            {match.fine_match_inliers != null ? (
                              <p className="text-xs text-muted-foreground">
                                Fine match inliers {match.fine_match_inliers}
                              </p>
                            ) : null}
                            <p className="text-xs text-muted-foreground">
                              Bounds {match.west.toFixed(5)},{" "}
                              {match.south.toFixed(5)} to{" "}
                              {match.east.toFixed(5)}, {match.north.toFixed(5)}
                            </p>
                          </div>
                          <p className="text-sm tabular-nums text-muted-foreground">
                            {match.score.toFixed(5)}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="rounded-md border border-dashed p-8 text-center text-sm text-muted-foreground">
                    Search results will appear here after the first query.
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="live">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,420px)_1fr]">
            <Card>
              <CardHeader>
                <CardTitle>Live Drone Track</CardTitle>
                <CardDescription>
                  Upload drone footage. Scout extracts frames, runs VPS fixes,
                  and fuses optical-flow motion between corrections.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form className="grid gap-4" onSubmit={handleLive}>
                  <div className="grid gap-2">
                    <Label htmlFor="drone-video">Drone video</Label>
                    <Input
                      id="drone-video"
                      type="file"
                      accept="video/*"
                      onChange={(event) =>
                        setVideoFile(event.target.files?.[0] ?? null)
                      }
                    />
                  </div>
                  <Field
                    label="Altitude (m AGL)"
                    value={altitudeM}
                    onChange={setAltitudeM}
                  />
                  <Button type="submit" disabled={liveMutation.isPending}>
                    <Video />
                    {liveMutation.isPending ? "Processing..." : "Process Video"}
                  </Button>
                  {liveMutation.error ? (
                    <p className="text-sm text-destructive">
                      {liveMutation.error.message}
                    </p>
                  ) : null}
                </form>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Track Output</CardTitle>
                <CardDescription>
                  Fused lat/lon per extracted frame with VPS confidence.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {liveResult?.track.length ? (
                  <div className="space-y-4">
                    <div className="rounded-md border p-4">
                      <p className="text-sm text-muted-foreground">
                        Session {liveResult.session_id.slice(0, 8)} ·{" "}
                        {liveResult.status}
                      </p>
                      <p className="text-sm">
                        {liveResult.frame_count} frames ·{" "}
                        {liveResult.vps_fix_count} VPS fixes · median conf{" "}
                        {(liveResult.median_confidence * 100).toFixed(0)}%
                      </p>
                      <Button
                        className="mt-3"
                        type="button"
                        variant="outline"
                        onClick={downloadGeoJson}
                      >
                        Export GeoJSON
                      </Button>
                    </div>
                    <div className="max-h-[480px] space-y-2 overflow-y-auto">
                      {liveResult.track.map((point) => (
                        <div
                          className="grid gap-2 rounded-md border p-3 md:grid-cols-[1fr_auto]"
                          key={point.frame_id}
                        >
                          <div>
                            <p className="font-medium">{point.frame_id}</p>
                            <p className="text-sm">
                              {point.lat.toFixed(6)}, {point.lon.toFixed(6)}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {point.source} · conf{" "}
                              {(point.vps_confidence * 100).toFixed(0)}% · t=
                              {point.timestamp_s.toFixed(1)}s
                            </p>
                          </div>
                          <a
                            className="text-sm text-primary underline"
                            href={`https://www.google.com/maps?q=${point.lat},${point.lon}`}
                            rel="noreferrer"
                            target="_blank"
                          >
                            Map
                          </a>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="rounded-md border border-dashed p-8 text-center text-sm text-muted-foreground">
                    {liveResult?.error ??
                      "Upload a drone clip to generate a geolocated track."}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

function AuthImage({
  alt,
  className,
  src,
}: {
  alt: string
  className?: string
  src: string
}) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    let createdUrl: string | null = null
    scoutBlob(src).then((blob) => {
      if (!active) {
        return
      }
      createdUrl = URL.createObjectURL(blob)
      setObjectUrl(createdUrl)
    })
    return () => {
      active = false
      if (createdUrl) {
        URL.revokeObjectURL(createdUrl)
      }
    }
  }, [src])

  if (!objectUrl) {
    return <div className={`${className ?? ""} bg-muted`} />
  }

  return <img alt={alt} className={className} src={objectUrl} />
}

function MetricCard({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Database
  label: string
  value: number | string
}) {
  return (
    <Card className="gap-3 py-4">
      <CardContent className="flex items-center gap-3">
        <div className="flex size-10 items-center justify-center rounded-md border bg-muted">
          <Icon className="size-5 text-muted-foreground" />
        </div>
        <div>
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="text-2xl font-semibold tabular-nums">{value}</p>
        </div>
      </CardContent>
    </Card>
  )
}

function Field({
  label,
  value,
  onChange,
}: {
  label: string
  value: string
  onChange: (value: string) => void
}) {
  const id = label.toLowerCase().replace(/\s+/g, "-")
  return (
    <div className="grid gap-2">
      <Label htmlFor={id}>{label}</Label>
      <Input
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  )
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
