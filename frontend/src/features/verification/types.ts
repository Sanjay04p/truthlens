export type VerificationLabel = 'Real' | 'Fake' | 'Unverified' | 'Error'

export type SourceInfo = {
  name: string
  domain?: string
  url?: string
  role?: string
}

export type VerificationBreakdown = Record<string, number>

export type TextVerificationResult = {
  tier: number
  label: VerificationLabel
  confidence_score: string
  breakdown?: VerificationBreakdown
  reason: string
  sources?: SourceInfo[]
  verification_type?: string
  matched_claim?: string
}

export type ImageVerificationResult = {
  modality: 'image'
  label: VerificationLabel
  confidence_score?: string
  provider_used?: string
  breakdown?: VerificationBreakdown
  reason?: string
}

export type VerificationResult = TextVerificationResult | ImageVerificationResult

export type VerificationHistoryItem = {
  id: string
  title: string
  label: VerificationLabel
  kind: 'text' | 'image'
  result: VerificationResult
  createdAt: string
}

export function isTextResult(result: VerificationResult): result is TextVerificationResult {
  return 'tier' in result
}
