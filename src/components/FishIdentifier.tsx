import { useState, useRef } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload, Fish, Loader2, Camera, Info } from "lucide-react";
import { toast } from "sonner";
import { pipeline, env } from '@huggingface/transformers';

// Configure transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

interface IdentificationResult {
  label: string;
  score: number;
  species?: string;
  genus?: string;
  commonName?: string;
}

export function FishIdentifier() {
  const [image, setImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<IdentificationResult[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (file: File) => {
    if (!file.type.startsWith('image/')) {
      toast.error("Please upload a valid image file");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      setImage(e.target?.result as string);
      setResults([]);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      handleImageUpload(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleImageUpload(file);
    }
  };

  const identifyFish = async () => {
    if (!image) return;

    setIsLoading(true);
    try {
      // Load image classification pipeline with a general model
      // Note: In a production app, you'd want to use a fish-specific model
      const classifier = await pipeline(
        'image-classification',
        'google/vit-base-patch16-224',
        { device: 'webgpu' }
      );

      const result = await classifier(image);
      
      // Process results to extract fish-related information
      const processedResults = result.slice(0, 5).map((item: any) => {
        const label = item.label.toLowerCase();
        const isFishRelated = label.includes('fish') || 
                             label.includes('salmon') || 
                             label.includes('tuna') || 
                             label.includes('bass') || 
                             label.includes('shark') ||
                             label.includes('trout') ||
                             label.includes('cod') ||
                             label.includes('marine') ||
                             label.includes('aquatic');

        return {
          label: item.label,
          score: item.score,
          species: isFishRelated ? extractSpecies(item.label) : undefined,
          genus: isFishRelated ? extractGenus(item.label) : undefined,
          commonName: isFishRelated ? item.label : undefined
        };
      });

      setResults(processedResults);
      
      if (processedResults.every(r => !r.species)) {
        toast.info("This might not be a fish image. The AI couldn't detect any fish species.");
      }
    } catch (error) {
      console.error('Error identifying fish:', error);
      toast.error("Failed to identify fish. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const extractSpecies = (label: string): string => {
    // Simple species extraction - in production, use a proper fish taxonomy database
    const words = label.split(' ');
    return words.length >= 2 ? `${words[0]} ${words[1]}` : label;
  };

  const extractGenus = (label: string): string => {
    // Simple genus extraction
    return label.split(' ')[0];
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
          Upload a photo of any fish and discover its species, genus, and scientific classification using advanced AI
        </p>
      </div>

      {/* Upload Area */}
      <Card className="ocean-glow">
        <CardContent className="p-8">
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
              isDragOver
                ? 'border-primary bg-primary/5'
                : 'border-border hover:border-primary/50'
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
                        Analyzing...
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
                      setResults([]);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = '';
                      }
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
                  <h3 className="text-lg font-semibold mb-2">Upload a Fish Photo</h3>
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

      {/* Results */}
      {results.length > 0 && (
        <Card className="ocean-glow">
          <CardContent className="p-6">
            <div className="flex items-center space-x-2 mb-4">
              <Info className="h-5 w-5 text-primary" />
              <h2 className="text-xl font-semibold">Identification Results</h2>
            </div>
            <div className="space-y-4">
              {results.map((result, index) => (
                <div
                  key={index}
                  className="p-4 rounded-lg border bg-card/50 hover:bg-card transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-lg">{result.label}</h3>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                      result.score > 0.7 
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : result.score > 0.4
                        ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {(result.score * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                  {result.species && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-3 text-sm">
                      <div>
                        <span className="font-medium text-muted-foreground">Genus:</span>
                        <p className="italic">{result.genus}</p>
                      </div>
                      <div>
                        <span className="font-medium text-muted-foreground">Species:</span>
                        <p className="italic">{result.species}</p>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
              <p className="text-sm text-blue-700 dark:text-blue-300">
                <strong>Note:</strong> This identification is based on AI analysis and should be used for educational purposes. 
                For scientific research or critical identification, please consult with marine biology experts.
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}