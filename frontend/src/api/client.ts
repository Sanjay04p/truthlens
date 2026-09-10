import type { ImageVerificationResult, TextVerificationResult } from '../features/verification/types'

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
