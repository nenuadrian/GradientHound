declare global {
  interface Window {
    electronAPI: {
      openFile: () => Promise<string | null>
      openDirectory: () => Promise<string | null>
      getVersion: () => Promise<string>
    }
  }
}

export {}
