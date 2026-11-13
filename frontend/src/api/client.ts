import type { ImageVerificationResult, TextVerificationResult, VideoVerificationResult } from '../features/verification/types'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

type ApiError = { detail?: string }

async function parseError(response: Response): Promise<never> {
  let message = `Request failed with status ${response.status}`
  try {
    const body = await response.json() as ApiError
    if (body.detail) message = body.detail
  } catch {
    // Keep the status-based message when the API does not return JSON.
  }
  throw new Error(message)
}

export async function verifyText(text: string): Promise<TextVerificationResult> {
  const response = await fetch(`${API_BASE_URL}/verify-text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })

  if (!response.ok) return parseError(response)
  return response.json() as Promise<TextVerificationResult>
}

export async function verifyImage(file: File): Promise<ImageVerificationResult> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/verify-image`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) return parseError(response)
  return response.json() as Promise<ImageVerificationResult>
}

export async function verifyVideo(file: File): Promise<VideoVerificationResult> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/verify-video`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) return parseError(response)

  const body = await response.json() as {
    verdict: string
    confidence?: number
    explanation?: string
    strategy?: string
    metrics?: Record<string, number>
  }
  const isReal = body.verdict.toLowerCase().includes('real') || body.verdict.toLowerCase().includes('authentic')
  const label = isReal ? 'Real' : body.verdict.toLowerCase() === 'error' ? 'Error' : 'Fake'

  return {
    modality: 'video',
    label,
    confidence_score: `${Math.round(body.confidence ?? 0)}%`,
    provider_used: body.strategy || 'Video analysis',
    breakdown: body.metrics || {},
    reason: body.explanation || body.verdict,
  }
}
