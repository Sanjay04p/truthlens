import { useState } from 'react'
import { verifyImage, verifyText } from '../../../api/client'
import type { VerificationResult } from '../types'

export function useVerification() {
  const [result, setResult] = useState<VerificationResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function run(request: () => Promise<VerificationResult>) {
    setIsLoading(true)
    setError(null)
    try {
      const nextResult = await request()
      setResult(nextResult)
      return nextResult
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : 'Something went wrong while verifying.'
      setError(message)
      return null
    } finally {
      setIsLoading(false)
    }
  }

  return {
    result,
    isLoading,
    error,
    verifyText: (text: string) => run(() => verifyText(text)),
    verifyImage: (file: File) => run(() => verifyImage(file)),
    selectResult: (nextResult: VerificationResult) => { setResult(nextResult); setError(null) },
    clearResult: () => { setResult(null); setError(null) },
  }
}
