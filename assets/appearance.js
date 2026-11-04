export default function (component) {
  const { parentElement, data, setStateValue } = component;
  const storageKey = "shopper-signals.appearance";
  const preferences = new Set(["system", "light", "dark"]);
  const root = document.documentElement;
  const select = parentElement.querySelector("select");
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  let preference = data?.appearance?.preference || "system";
  let acknowledged = JSON.stringify(data?.appearance || null);

  const readPreference = () => {
    try {
      const stored = window.localStorage.getItem(storageKey);
      preference = preferences.has(stored) ? stored : "system";
    } catch {
      // Restricted storage still permits a preference for this open session.
      if (!preferences.has(preference)) preference = "system";
    }
  };

  const applyAppearance = () => {
    const effective = preference === "system"
      ? (media.matches ? "dark" : "light")
      : preference;
    root.dataset.shopperTheme = effective;
    root.style.colorScheme = effective;
    select.value = preference;
    const state = { preference, effective };
    const encoded = JSON.stringify(state);
    if (encoded !== acknowledged) {
      acknowledged = encoded;
      setStateValue("appearance", state);
    }
  };

  const onPreferenceChange = () => {
    preference = select.value;
    try {
      window.localStorage.setItem(storageKey, preference);
    } catch {
      // Browser storage can be disabled without disabling the control.
    }
    applyAppearance();
  };

  const onStorageChange = (event) => {
    if (event.key === storageKey || event.key === null) {
      readPreference();
      applyAppearance();
    }
  };

  readPreference();
  select.addEventListener("change", onPreferenceChange);
  media.addEventListener("change", applyAppearance);
  window.addEventListener("storage", onStorageChange);
  applyAppearance();

  let observer = null;
  let focusTimeout = null;
  const focus = data?.focus_request;
  const focusTarget = () => {
    if (!focus || root.dataset.shopperFocusToken === focus.token) return true;
    let target;
    try {
      target = document.querySelector(focus.selector);
    } catch {
      return true;
    }
    if (!target || !target.isConnected || !target.getClientRects().length) return false;
    if (getComputedStyle(target).visibility !== "visible" || target.matches(":disabled") || target.closest("[inert]")) return false;
    // Native buttons must keep their normal place in sequential Tab navigation.
    if (target.tabIndex < 0 && !target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
    target.focus({ preventScroll: true });
    if (document.activeElement !== target) return false;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    root.dataset.shopperFocusToken = focus.token;
    return true;
  };

  if (focus && typeof focus.token === "string" && typeof focus.selector === "string" && !focusTarget()) {
    observer = new MutationObserver(() => {
      if (focusTarget()) {
        observer.disconnect();
        window.clearTimeout(focusTimeout);
      }
    });
    observer.observe(document.body, {
      childList: true, subtree: true, attributes: true,
      attributeFilter: ["class", "style", "hidden", "disabled", "inert", "data-test-script-state"],
    });
    focusTimeout = window.setTimeout(() => observer.disconnect(), 10000);
  }

  return () => {
    select.removeEventListener("change", onPreferenceChange);
    media.removeEventListener("change", applyAppearance);
    window.removeEventListener("storage", onStorageChange);
    observer?.disconnect();
    window.clearTimeout(focusTimeout);
  };
}
