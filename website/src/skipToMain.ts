export const MAIN_CONTENT_ID = "main";

export type SkipActivationEvent = {
  preventDefault: () => void;
};

export type FocusableElement = {
  tabIndex: number;
  focus: () => void;
};

export function skipToMainContent(
  event: SkipActivationEvent,
  getElementById: (id: string) => FocusableElement | null,
): boolean {
  event.preventDefault();
  const main = getElementById(MAIN_CONTENT_ID);
  if (main === null) {
    return false;
  }
  main.tabIndex = -1;
  main.focus();
  return true;
}
