import { contextBridge, ipcRenderer } from 'electron'

const api = {
  openFile: (): Promise<string | null> => ipcRenderer.invoke('dialog:openFile'),
  openDirectory: (): Promise<string | null> => ipcRenderer.invoke('dialog:openDirectory'),
  getVersion: (): Promise<string> => ipcRenderer.invoke('app:getVersion')
}

contextBridge.exposeInMainWorld('electronAPI', api)
