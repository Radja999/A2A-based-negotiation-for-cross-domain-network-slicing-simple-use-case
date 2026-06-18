import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registry import A2A_BASE_URLS, base_url, card_url, rpc_url


class TestRegistryContents(unittest.TestCase):
    def test_all_agents_present(self):
        self.assertSetEqual(set(A2A_BASE_URLS.keys()), {"orchestrator", "ran", "edge"})

    def test_orchestrator_port(self):
        self.assertIn(":9000", A2A_BASE_URLS["orchestrator"])

    def test_ran_port(self):
        self.assertIn(":9001", A2A_BASE_URLS["ran"])

    def test_edge_port(self):
        self.assertIn(":9002", A2A_BASE_URLS["edge"])

    def test_urls_are_http(self):
        for name, url in A2A_BASE_URLS.items():
            with self.subTest(agent=name):
                self.assertTrue(url.startswith("http://"), f"{name} URL must start with http://")

    def test_urls_use_loopback(self):
        for name, url in A2A_BASE_URLS.items():
            with self.subTest(agent=name):
                self.assertTrue(
                    "127.0.0.1" in url or "localhost" in url,
                    f"{name} URL must use loopback address"
                )


class TestBaseUrlLookup(unittest.TestCase):
    def test_lookup_orchestrator(self):
        url = base_url("orchestrator")
        self.assertIn(":9000", url)

    def test_lookup_ran(self):
        url = base_url("ran")
        self.assertIn(":9001", url)

    def test_lookup_edge(self):
        url = base_url("edge")
        self.assertIn(":9002", url)

    def test_unknown_agent_raises_key_error(self):
        with self.assertRaises(KeyError):
            base_url("unknown_agent")

    def test_case_sensitive(self):
        with self.assertRaises(KeyError):
            base_url("RAN")


class TestCardAndRpcUrls(unittest.TestCase):
    def test_card_url_well_known_path(self):
        for name in ("orchestrator", "ran", "edge"):
            with self.subTest(agent=name):
                url = card_url(name)
                self.assertTrue(
                    url.endswith("/.well-known/agent-card.json"),
                    f"card_url({name!r}) must end with /.well-known/agent-card.json, got {url!r}"
                )

    def test_card_url_contains_base(self):
        self.assertTrue(card_url("ran").startswith(base_url("ran")))

    def test_rpc_url_ends_with_slash(self):
        for name in ("orchestrator", "ran", "edge"):
            with self.subTest(agent=name):
                self.assertTrue(rpc_url(name).endswith("/"))

    def test_rpc_url_contains_base(self):
        self.assertTrue(rpc_url("edge").startswith(base_url("edge")))

    def test_card_and_rpc_urls_differ(self):
        self.assertNotEqual(card_url("ran"), rpc_url("ran"))


if __name__ == "__main__":
    unittest.main()
