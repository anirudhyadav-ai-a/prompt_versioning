jest.mock('vscode');

import * as vscode from 'vscode';
import { callLLM, callLLMJson } from './client';

describe('LLM client', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('calls Copilot model and returns text', async () => {
        const mockModel = {
            id: 'copilot-claude-3.5-sonnet',
            family: 'claude',
            sendRequest: jest.fn().mockResolvedValue({
                text: (async function* () {
                    yield 'Hello ';
                    yield 'World';
                })(),
            }),
        };
        (vscode.lm.selectChatModels as jest.Mock).mockResolvedValue([mockModel]);

        const result = await callLLM('test prompt', 'system prompt');
        expect(result.text).toBe('Hello World');
        expect(mockModel.sendRequest).toHaveBeenCalledTimes(1);
    });

    it('throws when no models available', async () => {
        (vscode.lm.selectChatModels as jest.Mock).mockResolvedValue([]);
        await expect(callLLM('test', 'system')).rejects.toThrow('No Copilot LLM models available');
    });

    it('callLLMJson parses JSON response', async () => {
        const mockModel = {
            id: 'copilot-gpt-4o',
            family: 'gpt',
            sendRequest: jest.fn().mockResolvedValue({
                text: (async function* () {
                    yield '{"key": "value"}';
                })(),
            }),
        };
        (vscode.lm.selectChatModels as jest.Mock).mockResolvedValue([mockModel]);

        const result = await callLLMJson<{ key: string }>('test', 'system');
        expect(result.key).toBe('value');
    });

    it('callLLMJson strips markdown code fences', async () => {
        const mockModel = {
            id: 'copilot-claude',
            family: 'claude',
            sendRequest: jest.fn().mockResolvedValue({
                text: (async function* () {
                    yield '```json\n{"key": "value"}\n```';
                })(),
            }),
        };
        (vscode.lm.selectChatModels as jest.Mock).mockResolvedValue([mockModel]);

        const result = await callLLMJson<{ key: string }>('test', 'system');
        expect(result.key).toBe('value');
    });
});
