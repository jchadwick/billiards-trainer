import { test, expect } from '@playwright/test';

test('WebSocket diagnostic page', async ({ page }) => {
  // Listen for console messages
  page.on('console', msg => {
    console.log(`[${msg.type()}] ${msg.text()}`);
  });

  // Listen for page errors
  page.on('pageerror', error => {
    console.error('Page error:', error);
  });

  // Go to the WebSocket events page
  console.log('Navigating to WebSocket events page...');
  await page.goto('http://localhost:3000/websocket-events');

  // Wait a bit for the page to load
  await page.waitForTimeout(5000);

  // Take a screenshot
  await page.screenshot({ path: '/tmp/websocket-diag.png', fullPage: true });
  console.log('Screenshot saved to /tmp/websocket-diag.png');

  // Get the connection state
  const connectionState = await page.locator('text=/CONNECTED|CONNECTING|DISCONNECTED|ERROR|RECONNECTING/i').first().textContent();
  console.log('Connection state:', connectionState);

  // Check for any error messages
  const pageContent = await page.content();
  console.log('Page loaded successfully');

  // Wait longer to see if events come in
  await page.waitForTimeout(10000);

  // Take another screenshot
  await page.screenshot({ path: '/tmp/websocket-diag-after.png', fullPage: true });
});
