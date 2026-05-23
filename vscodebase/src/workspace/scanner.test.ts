jest.mock('vscode');

import * as vscode from 'vscode';
import { scanWorkspace, buildWorkspaceContext, getFileContext } from './scanner';

describe('workspace scanner', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('returns empty context when no files found', async () => {
        (vscode.workspace.findFiles as jest.Mock).mockResolvedValue([]);
        const ctx = await scanWorkspace();
        expect(ctx.files).toHaveLength(0);
        expect(ctx.summary).toContain('Total symbols: 0');
    });

    it('scans workspace files and extracts symbols', async () => {
        const mockUri = { fsPath: '/mock/file.py', scheme: 'file' };
        (vscode.workspace.findFiles as jest.Mock).mockResolvedValue([mockUri]);
        (vscode.workspace.openTextDocument as jest.Mock).mockResolvedValue({
            getText: () => 'class Foo:\n    pass\ndef bar():\n    pass',
            languageId: 'python',
        });
        (vscode.workspace as unknown as { asRelativePath: jest.Mock }).asRelativePath = jest.fn(() => 'file.py');

        const ctx = await scanWorkspace();
        expect(ctx.files.length).toBeGreaterThanOrEqual(1);
    });

    it('buildWorkspaceContext returns string with overview', async () => {
        (vscode.workspace.findFiles as jest.Mock).mockResolvedValue([]);
        const context = await buildWorkspaceContext();
        expect(context).toContain('No workspace files found');
    });

    it('getFileContext reads a file', async () => {
        const mockUri = { fsPath: '/mock/test.ts', scheme: 'file' };
        (vscode.workspace.openTextDocument as jest.Mock).mockResolvedValue({
            getText: () => 'function hello() {}',
            languageId: 'typescript',
        });
        (vscode.workspace as unknown as { asRelativePath: jest.Mock }).asRelativePath = jest.fn(() => 'test.ts');

        const info = await getFileContext(mockUri as vscode.Uri);
        expect(info.path).toBe('test.ts');
        expect(info.language).toBe('typescript');
    });
});
