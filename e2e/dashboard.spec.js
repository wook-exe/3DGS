import { expect, test } from "@playwright/test";

test("dashboard renders charts and release-control APIs", async ({ page }) => {
  await page.goto("/1.html");

  await expect(page.getByRole("heading", { name: /Project DORA Metrics Dashboard/ })).toBeVisible();
  await expect(page.locator("#releaseControls")).toContainText("Release Controls");
  await expect(page.locator("canvas")).toHaveCount(4);

  const flagsResponse = await page.request.get("/flags?user_id=e2e-student");
  expect(flagsResponse.ok()).toBeTruthy();
  const flags = await flagsResponse.json();
  expect(flags.features.model_status_sidebar).toBe(true);
  expect(flags.experiments).toHaveProperty("dashboard_chart_density");

  const eventResponse = await page.request.post("/events", {
    data: {
      event_name: "e2e_dashboard_loaded",
      user_id: "e2e-student",
      experiment_key: "dashboard_chart_density",
      properties: {
        source: "playwright"
      }
    }
  });
  expect(eventResponse.status()).toBe(201);
});
