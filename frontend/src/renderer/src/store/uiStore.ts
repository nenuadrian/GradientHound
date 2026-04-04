import { create } from 'zustand'

interface UIState {
  sidePanelOpen: boolean
  searchOpen: boolean
  searchQuery: string
  searchResults: string[]

  toggleSidePanel: () => void
  setSearchOpen: (open: boolean) => void
  setSearchQuery: (query: string) => void
  setSearchResults: (results: string[]) => void
}

export const useUIStore = create<UIState>((set) => ({
  sidePanelOpen: true,
  searchOpen: false,
  searchQuery: '',
  searchResults: [],

  toggleSidePanel: () => set(s => ({ sidePanelOpen: !s.sidePanelOpen })),
  setSearchOpen: (open) => set({ searchOpen: open }),
  setSearchQuery: (query) => set({ searchQuery: query }),
  setSearchResults: (results) => set({ searchResults: results })
}))
