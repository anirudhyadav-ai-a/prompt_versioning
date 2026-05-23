jest.mock('vscode');

import * as vscode from 'vscode';
import { activate, deactivate } from './extension';

describe('prompt_versioning extension', () => {
    let context: vscode.ExtensionContext;

    beforeEach(() => {
        jest.clearAllMocks();
        context = {
            subscriptions: [],
            extensionPath: '/mock/ext',
        } as unknown as vscode.ExtensionContext;
    });

    it('activates without errors', () => {
        expect(() => activate(context)).not.toThrow();
    });

    it('registers 3 commands', () => {
        activate(context);
        expect(vscode.commands.registerCommand).toHaveBeenCalledTimes(3);
        expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
            'prompts.register',
            expect.any(Function),
        );
        expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
            'prompts.drift',
            expect.any(Function),
        );
        expect(vscode.commands.registerCommand).toHaveBeenCalledWith(
            'prompts.test',
            expect.any(Function),
        );
    });

    it('registers chat participant', () => {
        activate(context);
        expect(vscode.chat.createChatParticipant).toHaveBeenCalledWith(
            'prompts.assistant',
            expect.any(Function),
        );
    });

    it('deactivates without errors', () => {
        expect(() => deactivate()).not.toThrow();
    });
});
