/* eslint-disable @typescript-eslint/no-explicit-any */

export const workspace = {
  getConfiguration: jest.fn(() => ({ get: jest.fn() })),
  findFiles: jest.fn(() => Promise.resolve([])),
  openTextDocument: jest.fn(() =>
    Promise.resolve({ getText: () => '', uri: { fsPath: '/mock/file.ts' } })
  ),
  workspaceFolders: [{ uri: { fsPath: '/mock/workspace' } }],
  onDidSaveTextDocument: jest.fn(),
  fs: { readFile: jest.fn(() => Promise.resolve(Buffer.from(''))) },
}

export const window = {
  activeTextEditor: undefined,
  showInformationMessage: jest.fn(),
  showErrorMessage: jest.fn(),
  showWarningMessage: jest.fn(),
  createOutputChannel: jest.fn(() => ({ appendLine: jest.fn(), show: jest.fn() })),
  registerTreeDataProvider: jest.fn(),
  withProgress: jest.fn((_opts: any, task: any) => task({ report: jest.fn() })),
}

export const commands = {
  registerCommand: jest.fn(),
}

export const lm = {
  selectChatModels: jest.fn(() =>
    Promise.resolve([
      {
        id: 'copilot-claude-3.5-sonnet',
        family: 'claude',
        sendRequest: jest.fn(() =>
          Promise.resolve({
            text: (async function* () { yield '{"result": "mock"}' })(),
          })
        ),
      },
    ])
  ),
}

export const chat = {
  createChatParticipant: jest.fn(() => ({ iconPath: null })),
}

export const LanguageModelChatMessage = {
  System: (content: string) => ({ role: 'system', content }),
  User: (content: string) => ({ role: 'user', content }),
}

export const LanguageModelError = class extends Error {
  code: string
  constructor(message: string, code: string) {
    super(message)
    this.code = code
  }
}

export const ThemeIcon = class { constructor(public id: string) {} }
export const Uri = {
  file: (path: string) => ({ fsPath: path, scheme: 'file', path }),
  parse: (str: string) => ({ fsPath: str, scheme: 'file', path: str }),
}
export const RelativePattern = class {
  constructor(public base: any, public pattern: string) {}
}
export const CancellationTokenSource = class {
  token = { isCancellationRequested: false }
  cancel() {}
  dispose() {}
}
export const ProgressLocation = { Notification: 15 }
export const TreeItem = class {
  label: string
  constructor(label: string) { this.label = label }
}
export const TreeItemCollapsibleState = { None: 0, Collapsed: 1, Expanded: 2 }
export enum ViewColumn { One = 1, Two = 2 }
