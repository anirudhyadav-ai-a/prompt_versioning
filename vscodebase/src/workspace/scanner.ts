import * as vscode from 'vscode';

export interface WorkspaceContext {
    files: FileInfo[];
    summary: string;
}

export interface FileInfo {
    path: string;
    language: string;
    symbols: string[];
    snippet: string;
}

const MAX_FILES = 50;
const MAX_SNIPPET_CHARS = 500;

/**
 * Scan the workspace for relevant source files and build a context summary.
 *
 * This is the "native Copilot behaviour" — instead of blindly following
 * an MD template, we actually read the codebase and inject its structure
 * into the LLM prompt.
 */
export async function scanWorkspace(
    fileGlob = '**/*.{ts,js,py,java,go,rs,cs,rb}',
): Promise<WorkspaceContext> {
    const uris = await vscode.workspace.findFiles(
        fileGlob,
        '**/node_modules/**',
        MAX_FILES,
    );

    const files: FileInfo[] = [];

    for (const uri of uris) {
        try {
            const doc = await vscode.workspace.openTextDocument(uri);
            const text = doc.getText();
            const language = doc.languageId;
            const relativePath = vscode.workspace.asRelativePath(uri);

            const symbols = extractSymbols(text, language);
            const snippet =
                text.length > MAX_SNIPPET_CHARS
                    ? text.slice(0, MAX_SNIPPET_CHARS) + '\n// ...'
                    : text;

            files.push({ path: relativePath, language, symbols, snippet });
        } catch {
            // skip files that can't be opened
        }
    }

    const summary = buildSummary(files);
    return { files, summary };
}

/**
 * Scan workspace and return a compact context string for LLM prompts.
 */
export async function buildWorkspaceContext(
    fileGlob?: string,
): Promise<string> {
    const ctx = await scanWorkspace(fileGlob);
    if (ctx.files.length === 0) {
        return '(No workspace files found)';
    }

    const parts = [
        `## Workspace Overview (${ctx.files.length} files)\n`,
        ctx.summary,
        '\n## File Details\n',
    ];

    for (const f of ctx.files.slice(0, 20)) {
        parts.push(`### ${f.path} (${f.language})`);
        if (f.symbols.length > 0) {
            parts.push(`Symbols: ${f.symbols.join(', ')}`);
        }
        parts.push('```' + f.language);
        parts.push(f.snippet);
        parts.push('```\n');
    }

    return parts.join('\n');
}

/**
 * Get context for a specific file (active editor or referenced).
 */
export async function getFileContext(
    uri: vscode.Uri,
): Promise<FileInfo> {
    const doc = await vscode.workspace.openTextDocument(uri);
    const text = doc.getText();
    const language = doc.languageId;
    const relativePath = vscode.workspace.asRelativePath(uri);
    const symbols = extractSymbols(text, language);

    return {
        path: relativePath,
        language,
        symbols,
        snippet: text,
    };
}

function extractSymbols(text: string, language: string): string[] {
    const symbols: string[] = [];
    const lines = text.split('\n');

    for (const line of lines) {
        let match: RegExpMatchArray | null;

        if (language === 'python') {
            match = line.match(/^(?:class|def|async def)\s+(\w+)/);
        } else if (['typescript', 'javascript'].includes(language)) {
            match = line.match(
                /(?:export\s+)?(?:class|function|const|interface|type|enum)\s+(\w+)/,
            );
        } else if (language === 'java' || language === 'csharp') {
            match = line.match(
                /(?:public|private|protected)?\s*(?:class|interface|enum|record)\s+(\w+)/,
            );
        } else if (language === 'go') {
            match = line.match(/^(?:func|type)\s+(\w+)/);
        } else if (language === 'rust') {
            match = line.match(
                /^(?:pub\s+)?(?:fn|struct|enum|trait|impl|type)\s+(\w+)/,
            );
        } else {
            match = line.match(
                /(?:class|function|def|fn|struct|interface)\s+(\w+)/,
            );
        }

        if (match) {
            symbols.push(match[1]);
        }
    }

    return symbols;
}

function buildSummary(files: FileInfo[]): string {
    const langCounts: Record<string, number> = {};
    let totalSymbols = 0;

    for (const f of files) {
        langCounts[f.language] = (langCounts[f.language] ?? 0) + 1;
        totalSymbols += f.symbols.length;
    }

    const langList = Object.entries(langCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([lang, count]) => `${lang}: ${count}`)
        .join(', ');

    return `Languages: ${langList}\nTotal symbols: ${totalSymbols}`;
}
