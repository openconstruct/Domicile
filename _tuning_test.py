"""Cache invalidation, shared adapters, and tool registration regressions."""
import asyncio
import json
import os
import runpy
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import config_loader as CL
import credentials
import expression
import file_cache
import local_tools as LT


class CacheTests(unittest.TestCase):
    def test_settings_changes_and_caller_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('{"nested": {"value": 1}}')
            with patch.object(CL, '_load_json', wraps=CL._load_json) as load:
                data = CL._read_json(path)
                data['nested']['value'] = 99
                self.assertEqual(CL._read_json(path)['nested']['value'], 1)
                self.assertEqual(load.call_count, 1)
                path.write_text('{"nested": {"value": 2}}')
                self.assertEqual(CL._read_json(path)['nested']['value'], 2)
                other = path.with_suffix('.new')
                other.write_text('{"nested": {"value": 3}}')
                other.replace(path)
                self.assertEqual(CL._read_json(path)['nested']['value'], 3)
                path.unlink()
                self.assertEqual(CL._read_json(path), {})
                path.write_text('broken')
                self.assertEqual(CL._read_json(path), {})
                path.write_text('{"restored": true}')
                self.assertEqual(CL._read_json(path), {'restored': True})

    def test_health_checks_follow_configured_credentials(self):
        import selftest
        cfg = {'providers': {'fixture': {'adapter': 'openai_compatible', 'key_env': 'FIXTURE_API_KEY'}}}
        with patch.object(CL, 'load_config', return_value=cfg), \
                patch.object(credentials, 'get', return_value='fixture') as read:
            self.assertEqual(selftest.check_provider_keys().status, selftest.OK)
            read.return_value = ''
            self.assertEqual(selftest.check_provider_keys().status, selftest.FAIL)

    def test_credential_rotation_and_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            with patch.object(credentials, 'ENV_PATH', path):
                credentials.save({'EXAMPLE_API_KEY': 'first'})
                self.assertEqual(credentials.get('EXAMPLE_API_KEY'), 'first')
                credentials.values()['EXAMPLE_API_KEY'] = 'mutated'
                self.assertEqual(credentials.get('EXAMPLE_API_KEY'), 'first')
                credentials.save({'EXAMPLE_API_KEY': 'second'})
                self.assertEqual(credentials.get('EXAMPLE_API_KEY'), 'second')
                path.unlink()
                self.assertEqual(credentials.get('EXAMPLE_API_KEY'), '')

    def test_log_cache_refresh_and_exact_expiry(self):
        now = int(time.time())
        def row(ts, count):
            return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(ts)) + f',fixture,m,{count},2\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'usage.csv'
            path.write_text('ts_iso,provider,model,input,output\n'+row(now-86400, 3))
            with patch.object(expression, 'TOKEN_LOG', path), patch.object(expression.time, 'time', return_value=now) as clock:
                first = expression._token_series()
                self.assertEqual(first[1], {'fixture': [5]})
                with patch.object(Path, 'open', side_effect=AssertionError('Unchanged log reread')):
                    again = expression._token_series()
                again[1]['fixture'][0] = 999
                self.assertEqual(expression._token_series(), first)
                clock.return_value = now + 0.01
                self.assertIsNone(expression._token_series())
                with path.open('a') as handle:
                    handle.write(row(now, 8))
                self.assertEqual(expression._token_series()[1], {'fixture': [10]})
                path.write_text('')
                self.assertIsNone(expression._token_series())
                path.unlink()
                self.assertIsNone(expression._token_series())


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_registered_mcp_tools_validate_and_dispatch(self):
        with patch.dict(os.environ, {'DOMICILE_TOOL_PHASE': 'full'}):
            server = runpy.run_path(str(Path(__file__).parent / 'codex_tools_server.py'))['mcp']
        schemas = await server.list_tools()
        self.assertEqual({s.name for s in schemas}, set(LT.PHASE_FULL)-{'ReadImage'})
        for schema in schemas:
            values = {'string': 'fixture', 'array': ['fixture'], 'integer': 1,
                      'number': 1.0, 'boolean': False}
            args = {name: values[schema.input_schema['properties'][name]['type']]
                    for name in schema.input_schema.get('required', [])}
            with patch.object(LT, 'dispatch', new_callable=AsyncMock, return_value='ok') as dispatch:
                result = await server.call_tool(schema.name, args)
                self.assertFalse(result.is_error, schema.name)
                self.assertEqual(dispatch.await_args.args[0], schema.name)
                if schema.name == 'SetWakeup':
                    self.assertEqual(dispatch.await_args.args[1], {})
                if schema.name == 'WebSearch':
                    self.assertEqual(dispatch.await_args.args[1]['count'], 5)
        with patch.dict(os.environ, {'DOMICILE_TOOL_PHASE': 'none'}):
            server = runpy.run_path(str(Path(__file__).parent / 'codex_tools_server.py'))['mcp']
        self.assertEqual(await server.list_tools(), [])

    async def test_responses_runs_through_inherited_turn(self):
        import cchat
        from providers.openai_responses import OpenAIResponsesProvider
        provider = object.__new__(OpenAIResponsesProvider)
        provider.model = 'fixture'
        provider.max_tokens = 128
        async def events():
            yield NS(type='response.output_text.delta', delta='hel')
            yield NS(type='response.output_text.delta', delta='lo')
            yield NS(type='response.completed', response=NS(usage=None))
        provider._client = NS(default_headers={}, responses=NS(create=AsyncMock(return_value=events())))
        provider._tool_schemas = lambda: []
        provider._ensure_mcp = AsyncMock()
        provider._build_messages = lambda *args, **kw: []
        provider._max_rounds = lambda: 2
        with patch.object(cchat, 'domicile_hud', return_value=''):
            chat = {'wire': []}
            delta = Mock()
            self.assertEqual(await provider.astream_turn(chat, 'hello', on_delta=delta), ('hello', 'ok'))
            self.assertEqual([call.args[1] for call in delta.call_args_list], ['hel', 'hello', ''])
            self.assertEqual(chat['wire'][0]['content'], 'hello')

    async def test_chat_completion_stream_keeps_reasoning_and_cumulative_text(self):
        from providers.openai_compatible import OpenAICompatibleProvider
        provider = object.__new__(OpenAICompatibleProvider)
        provider.model = 'fixture'
        provider.max_tokens = 128
        provider._tool_schemas = lambda: []
        async def events():
            for content in ('hel', 'lo'):
                yield NS(usage=None, choices=[NS(finish_reason=None,
                    delta=NS(content=content, reasoning_content='', tool_calls=[]))])
        provider._client = NS(chat=NS(completions=NS(create=AsyncMock(return_value=events()))))
        delta = Mock()
        self.assertEqual(await provider._one_call([], delta, []), ('hello', [], None, {'reasoning_content': ''}))
        self.assertEqual([call.args[1] for call in delta.call_args_list], ['hel', 'hello'])

    async def test_incoming_turns_keep_sources_images_and_locking(self):
        import cchat
        for source, label in (("D", "discord"), ("C", "chat")):
            chat = {"title": "New chat", "messages": []}
            app = NS(turn_lock=asyncio.Lock(), _bridge_target=lambda: (chat, None), notify=Mock())
            def append(chat, role, text):
                self.assertTrue(app.turn_lock.locked())
                chat["messages"].append({"role": role, "text": text})
            with patch.object(cchat, 'append_message', side_effect=append), \
                    patch.object(cchat, 'run_turn', new_callable=AsyncMock, return_value=('reply', 'ok')) as run, \
                    patch.object(cchat, 'maybe_compact', new_callable=AsyncMock), \
                    patch.object(cchat, 'save_inbound_image', return_value=Path('image.png')):
                self.assertEqual(await cchat.CCChat._attached_turn(
                    app, 'hello', source=source, label=label, images=['image'] if source == 'D' else None), 'reply')
                self.assertEqual(run.await_args.kwargs['source'], source)
                self.assertEqual(chat['messages'][0]['text'], 'hello\n\n[image: image.png]' if source == 'D' else 'hello')
                self.assertEqual(chat['title'], chat['messages'][0]['text'])
                self.assertFalse(app.turn_lock.locked())
                app.notify.assert_called_once()

    async def test_background_notice_reads_only_the_log_tail(self):
        import cchat
        text = 'long log ' * 10000 + '🌳' * 5000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'fixture.log').write_text(text)
            ready = asyncio.Event()
            ready.set()
            bridge = NS(_ready=ready, send=AsyncMock())
            app = NS(bridge=bridge, turn_lock=asyncio.Lock(), _wakeup_target=lambda: ({'messages': []}, None))
            with patch.object(cchat, 'BGTASK_DIR', root), \
                    patch.object(cchat, 'append_message'), patch.object(cchat, 'log_wakeup'), \
                    patch.object(cchat, 'maybe_compact', new_callable=AsyncMock), \
                    patch.object(cchat, 'run_turn', new_callable=AsyncMock, return_value=('reply', 'ok')) as run:
                self.assertTrue(await cchat.CCChat._notify_bgtask(
                    app, {'id': 'fixture', 'exit_code': 0}, root / 'fixture.json'))
                self.assertIn('... [head trimmed] ...\n' + text[-4000:], run.await_args.args[2])

    async def test_claude_tool_wrappers_share_schema_and_dispatch(self):
        import cchat
        server = cchat._web_mcp_server()
        self.assertIsNotNone(server)
        # SDK server construction accepts the actual JSON schemas; calls still use shared execution.
        with patch.object(LT, 'execute', new_callable=AsyncMock, return_value={'content': 'fixture'}):
            self.assertEqual((await cchat._sdk_local_tool('WebSearch', {'query': 'q'}))['content'][0]['text'], 'fixture')


class ExtractionTests(unittest.TestCase):
    def test_both_adapters_preserve_retry_budget_and_inputs(self):
        from mem0_oai_llm import OpenAILLM
        from mem0_or_llm import OpenRouterLLM
        def response(text, reason='stop'):
            return NS(choices=[NS(message=NS(content=text), finish_reason=reason)])
        for cls in (OpenAILLM, OpenRouterLLM):
            llm = object.__new__(cls)
            llm.model = 'fixture'
            llm._request = Mock(side_effect=[response('', 'length'), response('{"facts": []}')])
            messages = [{'role': 'user', 'content': 'extract this'}]
            self.assertEqual(llm.generate_response(messages, {'type': 'json_object'}, max_tokens=100), '{"facts": []}')
            self.assertEqual([c.args[1] for c in llm._request.call_args_list], [4196, 12388])
            self.assertEqual(messages[0]['content'], 'extract this')
            llm._request = Mock(side_effect=RuntimeError('402 insufficient credit'))
            from mem0_llm_common import UsageLimitReached
            with self.assertRaises(UsageLimitReached): llm.generate_response(messages)
            llm._request.assert_called_once()
        llm = object.__new__(OpenAILLM)
        llm._request = Mock(side_effect=[response('not json'), response('{"facts": []}')])
        self.assertEqual(llm.generate_response(messages, {'type': 'json_object'}), '{"facts": []}')
        self.assertEqual(llm._request.call_count, 2)


if __name__ == '__main__':
    unittest.main()
