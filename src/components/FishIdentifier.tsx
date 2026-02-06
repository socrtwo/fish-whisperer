import { useState, useRef, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Upload, Fish, Loader2, Camera, Info, AlertTriangle } from "lucide-react";
import { toast } from "sonner";
import { pipeline, env } from "@huggingface/transformers";
import {
  FishSpecies,
  getCandidateLabels,
  findFishByLabel,
} from "@/data/fishDatabase";

// Configure transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

interface ClassificationMatch {
  fish: FishSpecies;
  score: number;
}

// Cache the classifier so it's only loaded once
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let classifierInstance: any = null;

export function FishIdentifier() {
  const [image, setImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState("");
  const [matches, setMatches] = useState<ClassificationMatch[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [noFishDetected, setNoFishDetected] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedFish =
    selectedId != null
      ? matches.find((m) => m.fish.id === selectedId) ?? matches[0]
      : matches[0];

  const isUncertain =
    matches.length > 1 &&
    selectedFish != null &&
    (selectedFish.score < 0.10 ||
      (matches.length >= 2 && matches[0].score / matches[1].score < 1.5));

  const handleImageUpload = (file: File) => {
    if (!file.type.startsWith("image/")) {
      toast.error("Please upload a valid image file");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target?.result as string);
      setMatches([]);
      setSelectedId(null);
      setNoFishDetected(false);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleImageUpload(file);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleImageUpload(file);
  };

  const getClassifier = useCallback(async () => {
    if (classifierInstance) return classifierInstance;

    setLoadingStatus("Downloading AI model (first time only)...");
    try {
      classifierInstance = await pipeline(
        "zero-shot-image-classification",
        "Xenova/clip-vit-base-patch32"
      );
    } catch {
      // WebGPU may not be available; fall back without specifying device
      classifierInstance = await pipeline(
        "zero-shot-image-classification",
        "Xenova/clip-vit-base-patch32"
      );
    }
    return classifierInstance;
  }, []);

  const identifyFish = async () => {
    if (!image) return;

    setIsLoading(true);
    setMatches([]);
    setSelectedId(null);
    setNoFishDetected(false);

    try {
      const classifier = await getClassifier();
      setLoadingStatus("Analyzing image...");

      const candidateLabels = getCandidateLabels();
      const results: { label: string; score: number }[] = await classifier(
        image,
        candidateLabels
      );

      // Separate fish results from non-fish results
      const fishResults: ClassificationMatch[] = [];
      let nonFishScore = 0;

      for (const r of results) {
        const fish = findFishByLabel(r.label);
        if (fish) {
          fishResults.push({ fish, score: r.score });
        } else {
          nonFishScore += r.score;
        }
      }

      // Sort by score descending
      fishResults.sort((a, b) => b.score - a.score);

      // If the combined non-fish labels outscore all fish labels, no fish detected
      const topFishScore = fishResults[0]?.score ?? 0;
      if (nonFishScore > topFishScore && topFishScore < 0.03) {
        setNoFishDetected(true);
        toast.info(
          "No fish detected in this image. Try a photo with a fish more clearly visible."
        );
      } else {
        // Take top 10 matches for the dropdown
        const topMatches = fishResults.slice(0, 10);
        setMatches(topMatches);
        setSelectedId(topMatches[0]?.fish.id ?? null);
      }
    } catch (error) {
      console.error("Error identifying fish:", error);
      toast.error("Failed to identify fish. Please try again.");
    } finally {
      setIsLoading(false);
      setLoadingStatus("");
    }
  };

  const confidenceColor = (score: number) => {
    if (score > 0.15)
      return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
    if (score > 0.06)
      return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200";
    return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
  };

  const confidenceLabel = (score: number) => {
    if (score > 0.15) return "High";
    if (score > 0.06) return "Medium";
    return "Low";
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center space-x-3">
          <Fish className="h-8 w-8 text-primary wave-animation" />
          <h1 className="text-4xl font-bold ocean-gradient bg-clip-text text-transparent">
            Fish Species Identifier
          </h1>
        </div>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Upload a photo of a fish &mdash; on a dock, in a boat, or in the water
          &mdash; and AI will identify the species
        </p>
      </div>

      {/* Upload Area */}
      <Card className="ocean-glow">
        <CardContent className="p-8">
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
              isDragOver
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/50"
            }`}
            onDrop={handleDrop}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
          >
            {image ? (
              <div className="space-y-4">
                <img
                  src={image}
                  alt="Uploaded fish"
                  className="max-w-full max-h-64 mx-auto rounded-lg shadow-lg"
                />
                <div className="flex flex-col sm:flex-row gap-3 justify-center">
                  <Button
                    onClick={identifyFish}
                    disabled={isLoading}
                    className="ocean-gradient hover:scale-105 transition-transform"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        {loadingStatus || "Analyzing..."}
                      </>
                    ) : (
                      <>
                        <Fish className="h-4 w-4 mr-2" />
                        Identify Fish Species
                      </>
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setImage(null);
                      setMatches([]);
                      setSelectedId(null);
                      setNoFishDetected(false);
                      if (fileInputRef.current) fileInputRef.current.value = "";
                    }}
                  >
                    Upload Different Image
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center">
                  <Camera className="h-8 w-8 text-primary" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-2">
                    Upload a Fish Photo
                  </h3>
                  <p className="text-muted-foreground mb-4">
                    Drag and drop your image here, or click to browse
                  </p>
                  <Button
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    Choose Image
                  </Button>
                </div>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        </CardContent>
      </Card>

      {/* No fish detected */}
      {noFishDetected && (
        <Card className="border-yellow-300 dark:border-yellow-700">
          <CardContent className="p-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-6 w-6 text-yellow-600 dark:text-yellow-400 shrink-0 mt-0.5" />
              <div>
                <h2 className="text-lg font-semibold mb-1">No Fish Detected</h2>
                <p className="text-muted-foreground">
                  The AI could not identify a fish in this image. For best
                  results, upload a clear photo where the fish is prominently
                  visible &mdash; on a dock, held up, on a boat deck, or in the
                  water.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {selectedFish && matches.length > 0 && (
        <Card className="ocean-glow">
          <CardContent className="p-6">
            <div className="flex items-center space-x-2 mb-4">
              <Info className="h-5 w-5 text-primary" />
              <h2 className="text-xl font-semibold">Identification Results</h2>
            </div>

            {/* Primary result */}
            <div className="p-5 rounded-lg border bg-card/50 space-y-4">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <h3 className="font-bold text-xl">
                  {selectedFish.fish.commonName}
                </h3>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${confidenceColor(selectedFish.score)}`}
                >
                  {confidenceLabel(selectedFish.score)} confidence &mdash;{" "}
                  {(selectedFish.score * 100).toFixed(1)}%
                </span>
              </div>

              <p className="text-muted-foreground">
                {selectedFish.fish.description}
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-muted-foreground">
                    Scientific Name:
                  </span>
                  <p className="italic">{selectedFish.fish.scientificName}</p>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">
                    Genus:
                  </span>
                  <p className="italic">{selectedFish.fish.genus}</p>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">
                    Family:
                  </span>
                  <p>{selectedFish.fish.family}</p>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">
                    Habitat:
                  </span>
                  <p className="capitalize">
                    {selectedFish.fish.habitat === "both"
                      ? "Freshwater & Saltwater"
                      : selectedFish.fish.habitat}
                  </p>
                </div>
              </div>
            </div>

            {/* Alternative species dropdown — shown when identification is uncertain */}
            {isUncertain && matches.length > 1 && (
              <div className="mt-5 p-4 rounded-lg border border-yellow-300 dark:border-yellow-700 bg-yellow-50/50 dark:bg-yellow-950/20 space-y-3">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-sm">
                      Identification is uncertain
                    </p>
                    <p className="text-sm text-muted-foreground">
                      The AI found several possible matches. Select the correct
                      species if you know it:
                    </p>
                  </div>
                </div>

                <Select
                  value={selectedId ?? undefined}
                  onValueChange={(value) => setSelectedId(value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select a species..." />
                  </SelectTrigger>
                  <SelectContent>
                    {matches.map((m) => (
                      <SelectItem key={m.fish.id} value={m.fish.id}>
                        {m.fish.commonName} &mdash;{" "}
                        {(m.score * 100).toFixed(1)}%
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* Other possible matches list (always shown when there are multiple) */}
            {matches.length > 1 && (
              <div className="mt-5">
                <h4 className="font-medium text-sm text-muted-foreground mb-3">
                  Other possible matches
                </h4>
                <div className="space-y-2">
                  {matches
                    .filter((m) => m.fish.id !== selectedFish.fish.id)
                    .slice(0, 4)
                    .map((m) => (
                      <button
                        key={m.fish.id}
                        className="w-full flex items-center justify-between p-3 rounded-lg border bg-card/30 hover:bg-card/80 transition-colors text-left"
                        onClick={() => setSelectedId(m.fish.id)}
                      >
                        <div>
                          <span className="font-medium">{m.fish.commonName}</span>
                          <span className="text-muted-foreground text-sm ml-2 italic">
                            {m.fish.scientificName}
                          </span>
                        </div>
                        <span
                          className={`px-2 py-0.5 rounded-full text-xs font-medium ${confidenceColor(m.score)}`}
                        >
                          {(m.score * 100).toFixed(1)}%
                        </span>
                      </button>
                    ))}
                </div>
              </div>
            )}

            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
              <p className="text-sm text-blue-700 dark:text-blue-300">
                <strong>Note:</strong> This identification is based on AI
                analysis and should be used for educational purposes. For
                scientific research or critical identification, please consult
                with marine biology experts.
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
