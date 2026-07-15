Traceback (most recent call last):
  File "/Users/joshuazyzak/delphi-bench/.venv/bin/delphi", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/Users/joshuazyzak/delphi-bench/common/cli.py", line 826, in main
    eval_context = _default_eval_context(args.suite)  # pragma: no cover - wired suite
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/joshuazyzak/delphi-bench/common/cli.py", line 520, in _default_eval_context
    records = MetaculusFetcher(http=http, secrets=EnvSecretProvider()).fetch(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/joshuazyzak/delphi-bench/benchmarks/fetchers/metaculus_api.py", line 192, in fetch
    payload = self._http.get_json(url, params=query, headers=headers)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/joshuazyzak/delphi-bench/common/http/client.py", line 155, in get_json
    response = self._request_with_retry("GET", url, params=params, headers=headers)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/joshuazyzak/delphi-bench/common/http/client.py", line 128, in _request_with_retry
    for attempt in Retrying(
                   ^^^^^^^^^
  File "/Users/joshuazyzak/delphi-bench/.venv/lib/python3.12/site-packages/tenacity/__init__.py", line 438, in __iter__
    do = self.iter(retry_state=retry_state)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/joshuazyzak/delphi-bench/.venv/lib/python3.12/site-packages/tenacity/__init__.py", line 371, in iter
    result = action(retry_state)
             ^^^^^^^^^^^^^^^^^^^
  File "/Users/joshuazyzak/delphi-bench/.venv/lib/python3.12/site-packages/tenacity/__init__.py", line 393, in <lambda>
    self._add_action_func(lambda rs: rs.outcome.result())
                                     ^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.12/concurrent/futures/_base.py", line 449, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/anaconda3/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
  File "/Users/joshuazyzak/delphi-bench/common/http/client.py", line 136, in _request_with_retry
    return self._raw_request(method, url, params=params, headers=headers, json=json)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/joshuazyzak/delphi-bench/common/http/client.py", line 116, in _raw_request
    return self._check_status(url, response)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/joshuazyzak/delphi-bench/common/http/client.py", line 99, in _check_status
    raise HttpError(f"{status} client error for {url}: {response.text[:200]!r}")
common.http.errors.HttpError: 403 client error for https://www.metaculus.com/api/posts/: 'Permission Error: The API is only available to authenticated users. Please create an account and use your API token to access the API.'
