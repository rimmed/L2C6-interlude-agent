/*
 * Copyright (c) 2013 L2jMobius
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
 * IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.l2jmobius.gameserver.config.custom;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import org.l2jmobius.commons.util.ConfigReader;
import org.l2jmobius.gameserver.model.Location;
import org.l2jmobius.gameserver.model.item.holders.ItemHolder;
import org.l2jmobius.gameserver.model.item.type.CrystalType;

/**
 * This class loads all the RL agent environment related configurations.
 */
public class RLAgentEnvironmentConfig
{
	private static final Logger LOGGER = Logger.getLogger(RLAgentEnvironmentConfig.class.getName());
	
	// File
	private static final String RL_AGENT_ENVIRONMENT_CONFIG_FILE = "./config/Custom/RLAgentEnvironment.ini";
	
	/**
	 * Farm zone of a level bracket.
	 * @param minLevel first level of the bracket
	 * @param maxLevel last level of the bracket
	 * @param anchors teleport anchors, empty when {@code useStartPoint} is set
	 * @param useStartPoint {@code true} to use the race's newbie start point
	 */
	public record FarmZone(int minLevel, int maxLevel, List<Location> anchors, boolean useStartPoint)
	{
	}
	
	/**
	 * One buff of a buff set.
	 * @param skillId the buff skill
	 * @param minLevel first agent level that gets it
	 * @param maxLevel last agent level that gets it
	 * @param mageOnly only for mages
	 * @param fighterOnly only for fighters
	 */
	public record BuffEntry(int skillId, int minLevel, int maxLevel, boolean mageOnly, boolean fighterOnly)
	{
	}
	
	/**
	 * Buff potions given to agents of a level range.
	 * @param itemId the potion
	 * @param count how many an agent keeps
	 * @param minLevel first agent level that gets it
	 * @param maxLevel last agent level that gets it
	 * @param mageOnly only for mages
	 * @param fighterOnly only for fighters
	 */
	public record PotionEntry(int itemId, int count, int minLevel, int maxLevel, boolean mageOnly, boolean fighterOnly)
	{
	}
	
	/** Support classes whose buffs form the {@code support} buff set ({@code RLBuffSupport.<name>}). */
	public static final String[] BUFF_SUPPORT_CLASSES =
	{
		"Prophet",
		"ElvenElder",
		"ShillienElder",
		"Bishop",
		"Warcryer",
		"Overlord"
	};
	
	// Constants
	public static boolean RL_AGENT_ENABLED;
	public static Pattern RL_AGENT_ACCOUNT_PATTERN;
	public static boolean RL_APPLY_TO_EXISTING_CHARACTERS;
	public static boolean RL_AUTO_CREATE_CHARACTER;
	public static List<String> RL_NAME_PREFIXES;
	public static int RL_NAME_SUFFIX_LENGTH;
	public static int RL_MIN_LEVEL;
	public static int RL_MAX_LEVEL;
	public static double RL_LEVEL_MEAN;
	public static double RL_LEVEL_STDDEV;
	public static boolean RL_AUTO_LEARN_SKILLS;
	public static boolean RL_AUTO_CLASS_TRANSFER;
	public static boolean RL_AUTO_EQUIP;
	/** Level at which each grade starts, ordered by level. */
	public static Map<Integer, CrystalType> RL_GRADE_LEVELS;
	public static int RL_SOULSHOT_COUNT;
	public static int RL_SPIRITSHOT_COUNT;
	public static int RL_ARROW_COUNT;
	public static boolean RL_DEATH_PENALTY;
	public static boolean RL_RESET_LEVEL_ON_LEVEL_UP;
	public static int RL_SKILL_ITEM_COUNT;
	public static int RL_BEAST_SHOT_COUNT;
	public static boolean RL_USE_BLESSED_SPIRITSHOTS;
	public static boolean RL_AUTO_ENABLE_SHOTS;
	public static List<ItemHolder> RL_CONSUMABLES;
	public static boolean RL_TELEPORT_TO_FARM_ZONE;
	public static boolean RL_TELEPORT_ON_LOGIN;
	public static List<FarmZone> RL_FARM_ZONES;
	public static int RL_FARM_ZONE_SPREAD;
	/** Pick farm spots from the monster spawns ({@code auto}) instead of the fixed {@link #RL_FARM_ZONES} anchors. */
	public static boolean RL_FARM_AUTO;
	public static int RL_FARM_LEVEL_BELOW;
	public static int RL_FARM_LEVEL_ABOVE;
	/** Percent of an area's monsters that must lie in the level window. */
	public static int RL_FARM_MIN_SHARE;
	public static int RL_FARM_MIN_MONSTERS;
	public static int RL_FARM_MOBS_PER_AGENT;
	public static List<String> RL_FARM_EXCLUDED_SPAWN_FOLDERS;
	/** Buff set name -> weight, in the configured order. */
	public static Map<String, Integer> RL_BUFF_SET_WEIGHTS;
	public static List<BuffEntry> RL_BUFF_NEWBIE;
	public static Map<String, List<BuffEntry>> RL_BUFF_SUPPORT;
	public static List<BuffEntry> RL_BUFF_DANCES_SONGS;
	public static List<BuffEntry> RL_BUFF_FULL;
	public static int RL_BUFF_REFRESH_INTERVAL;
	public static List<PotionEntry> RL_BUFF_POTIONS;
	/** Class family ({@code Prophet}, {@code Elder}, {@code Shaman}) -> build name -> weight. */
	public static Map<String, Map<String, Integer>> RL_BUILDS;
	public static int RL_UNDEAD_MIN_LEVEL;
	public static int RL_FARM_UNDEAD_LEVEL_BELOW;
	public static int RL_FARM_UNDEAD_MIN_SHARE;
	public static boolean RL_RESPAWN_ON_DEATH;
	public static int RL_RESPAWN_DELAY;
	
	public static void load()
	{
		final ConfigReader config = new ConfigReader(RL_AGENT_ENVIRONMENT_CONFIG_FILE);
		RL_AGENT_ENABLED = config.getBoolean("EnableRLAgentAutoSetup", false);
		RL_AGENT_ACCOUNT_PATTERN = Pattern.compile(config.getString("RLAgentAccountPattern", ".*"));
		RL_APPLY_TO_EXISTING_CHARACTERS = config.getBoolean("RLApplyToExistingCharacters", false);
		RL_AUTO_CREATE_CHARACTER = config.getBoolean("RLAutoCreateCharacter", true);
		RL_NAME_PREFIXES = new ArrayList<>();
		for (String prefix : config.getString("RLNamePrefixes", "Bot,Agent,RL").split(","))
		{
			final String clean = prefix.replaceAll("[^A-Za-z0-9]", "");
			if (!clean.isEmpty())
			{
				RL_NAME_PREFIXES.add(clean);
			}
		}
		if (RL_NAME_PREFIXES.isEmpty())
		{
			RL_NAME_PREFIXES.add("Bot");
		}
		RL_NAME_SUFFIX_LENGTH = Math.max(3, config.getInt("RLNameSuffixLength", 6));
		RL_MIN_LEVEL = Math.max(1, config.getInt("RLMinLevel", 1));
		RL_MAX_LEVEL = Math.max(RL_MIN_LEVEL, config.getInt("RLMaxLevel", 80));
		RL_LEVEL_MEAN = config.getDouble("RLLevelMean", 40);
		RL_LEVEL_STDDEV = Math.max(0, config.getDouble("RLLevelStdDev", 15));
		RL_AUTO_LEARN_SKILLS = config.getBoolean("RLAutoLearnSkills", true);
		RL_AUTO_CLASS_TRANSFER = config.getBoolean("RLAutoClassTransfer", true);
		RL_AUTO_EQUIP = config.getBoolean("RLAutoEquip", true);
		RL_GRADE_LEVELS = parseGradeLevels(config.getString("RLGradeLevels", "D:20;C:40;B:52;A:61;S:76"));
		RL_SOULSHOT_COUNT = config.getInt("RLSoulshotCount", 2000);
		RL_SPIRITSHOT_COUNT = config.getInt("RLSpiritshotCount", 1000);
		RL_ARROW_COUNT = config.getInt("RLArrowCount", 3000);
		RL_DEATH_PENALTY = config.getBoolean("RLAgentDeathPenalty", false);
		RL_RESET_LEVEL_ON_LEVEL_UP = config.getBoolean("RLResetLevelOnLevelUp", true);
		RL_SKILL_ITEM_COUNT = config.getInt("RLSkillItemCount", 100);
		RL_BEAST_SHOT_COUNT = config.getInt("RLBeastShotCount", 1000);
		RL_USE_BLESSED_SPIRITSHOTS = config.getBoolean("RLUseBlessedSpiritshots", true);
		RL_AUTO_ENABLE_SHOTS = config.getBoolean("RLAutoEnableShots", true);
		RL_CONSUMABLES = parseItems(config.getString("RLConsumables", "1061:50;1539:50"));
		RL_TELEPORT_TO_FARM_ZONE = config.getBoolean("RLTeleportToFarmZone", true);
		RL_TELEPORT_ON_LOGIN = config.getBoolean("RLTeleportOnLogin", true);
		RL_FARM_ZONES = parseFarmZones(config.getString("RLFarmZones", "1-80:start"));
		RL_FARM_ZONE_SPREAD = Math.max(0, config.getInt("RLFarmZoneSpread", 150));
		RL_FARM_AUTO = config.getString("RLFarmZoneMode", "auto").trim().equalsIgnoreCase("auto");
		RL_FARM_LEVEL_BELOW = Math.max(0, config.getInt("RLFarmLevelBelow", 2));
		RL_FARM_LEVEL_ABOVE = Math.max(0, config.getInt("RLFarmLevelAbove", 1));
		RL_FARM_MIN_SHARE = Math.min(100, Math.max(0, config.getInt("RLFarmMinShare", 70)));
		RL_FARM_MIN_MONSTERS = Math.max(1, config.getInt("RLFarmMinMonsters", 4));
		RL_FARM_MOBS_PER_AGENT = Math.max(1, config.getInt("RLFarmMobsPerAgent", 5));
		RL_FARM_EXCLUDED_SPAWN_FOLDERS = new ArrayList<>();
		for (String folder : config.getString("RLFarmExcludedSpawnFolders", "Castles,Catacombs,SevenSigns").split(","))
		{
			if (!folder.isBlank())
			{
				RL_FARM_EXCLUDED_SPAWN_FOLDERS.add(folder.trim());
			}
		}
		RL_BUFF_SET_WEIGHTS = parseWeights(config.getString("RLBuffSetWeights", "none:15;newbie:20;support:30;dances_songs:15;full:20"));
		RL_BUFF_NEWBIE = parseBuffs(config.getString("RLBuffNewbie", ""));
		RL_BUFF_SUPPORT = new LinkedHashMap<>();
		for (String supportClass : BUFF_SUPPORT_CLASSES)
		{
			final List<BuffEntry> buffs = parseBuffs(config.getString("RLBuffSupport." + supportClass, ""));
			if (!buffs.isEmpty())
			{
				RL_BUFF_SUPPORT.put(supportClass, buffs);
			}
		}
		RL_BUFF_DANCES_SONGS = parseBuffs(config.getString("RLBuffDancesSongs", ""));
		RL_BUFF_FULL = parseBuffs(config.getString("RLBuffFull", ""));
		RL_BUFF_REFRESH_INTERVAL = Math.max(0, config.getInt("RLBuffRefreshInterval", 30000));
		RL_BUFF_POTIONS = new ArrayList<>();
		for (BuffEntry entry : parseBuffs(config.getString("RLBuffPotions", "734:1-39,1374:40-80,735:1-39:fighter,1375:40-80:fighter,6035:1-39:mage,6036:40-80:mage")))
		{
			RL_BUFF_POTIONS.add(new PotionEntry(entry.skillId(), config.getInt("RLBuffPotionCount", 20), entry.minLevel(), entry.maxLevel(), entry.mageOnly(), entry.fighterOnly()));
		}
		RL_BUILDS = new LinkedHashMap<>();
		RL_BUILDS.put("Prophet", parseWeights(config.getString("RLBuilds.Prophet", "undead_mage:40;melee_heavy:30;melee_light:30")));
		RL_BUILDS.put("Elder", parseWeights(config.getString("RLBuilds.Elder", "undead_mage:70;melee_light:30")));
		RL_BUILDS.put("Shaman", parseWeights(config.getString("RLBuilds.Shaman", "caster:60;melee_heavy:40")));
		RL_UNDEAD_MIN_LEVEL = config.getInt("RLUndeadMinLevel", 30);
		RL_FARM_UNDEAD_LEVEL_BELOW = Math.max(0, config.getInt("RLFarmUndeadLevelBelow", 5));
		RL_FARM_UNDEAD_MIN_SHARE = Math.min(100, Math.max(0, config.getInt("RLFarmUndeadMinShare", 50)));
		RL_RESPAWN_ON_DEATH = config.getBoolean("RLRespawnOnDeath", true);
		RL_RESPAWN_DELAY = Math.max(0, config.getInt("RLRespawnDelay", 4000));
	}
	
	private static Map<Integer, CrystalType> parseGradeLevels(String value)
	{
		final Map<Integer, CrystalType> result = new TreeMap<>();
		for (String entry : value.split(";"))
		{
			final String[] parts = entry.trim().split(":");
			if (parts.length != 2)
			{
				continue;
			}
			try
			{
				result.put(Integer.parseInt(parts[1].trim()), CrystalType.valueOf(parts[0].trim().toUpperCase()));
			}
			catch (IllegalArgumentException e)
			{
				LOGGER.warning(RLAgentEnvironmentConfig.class.getSimpleName() + ": Invalid RLGradeLevels entry: " + entry);
			}
		}
		return Collections.unmodifiableMap(result);
	}
	
	private static List<ItemHolder> parseItems(String value)
	{
		final List<ItemHolder> result = new ArrayList<>();
		for (String entry : value.split(";"))
		{
			final String[] parts = entry.trim().split(":");
			if (parts.length != 2)
			{
				continue;
			}
			try
			{
				result.add(new ItemHolder(Integer.parseInt(parts[0].trim()), Integer.parseInt(parts[1].trim())));
			}
			catch (NumberFormatException e)
			{
				LOGGER.warning(RLAgentEnvironmentConfig.class.getSimpleName() + ": Invalid RLConsumables entry: " + entry);
			}
		}
		return Collections.unmodifiableList(result);
	}
	
	/**
	 * Parses {@code name:weight;name:weight} (names lower case, in the configured order).
	 * @param value the config value
	 * @return name -> weight
	 */
	private static Map<String, Integer> parseWeights(String value)
	{
		final Map<String, Integer> result = new LinkedHashMap<>();
		for (String entry : value.split(";"))
		{
			final String[] pair = entry.trim().split(":");
			if (pair.length != 2)
			{
				continue;
			}
			try
			{
				result.put(pair[0].trim().toLowerCase(), Math.max(0, Integer.parseInt(pair[1].trim())));
			}
			catch (NumberFormatException e)
			{
				LOGGER.warning(RLAgentEnvironmentConfig.class.getSimpleName() + ": Invalid weight entry: " + entry);
			}
		}
		return result;
	}
	
	/**
	 * Parses a buff list: {@code skillId[:minLevel-maxLevel][:mage|fighter]}, separated by commas or semicolons.
	 * @param value the config value
	 * @return the buffs in the configured order
	 */
	private static List<BuffEntry> parseBuffs(String value)
	{
		final List<BuffEntry> result = new ArrayList<>();
		for (String entry : value.split("[,;]"))
		{
			if (entry.isBlank())
			{
				continue;
			}
			try
			{
				final String[] parts = entry.trim().split(":");
				int minLevel = 1;
				int maxLevel = Integer.MAX_VALUE;
				boolean mageOnly = false;
				boolean fighterOnly = false;
				for (int i = 1; i < parts.length; i++)
				{
					final String part = parts[i].trim();
					if (part.equalsIgnoreCase("mage"))
					{
						mageOnly = true;
					}
					else if (part.equalsIgnoreCase("fighter"))
					{
						fighterOnly = true;
					}
					else
					{
						final String[] range = part.split("-");
						minLevel = Integer.parseInt(range[0].trim());
						maxLevel = Integer.parseInt(range[range.length - 1].trim());
					}
				}
				result.add(new BuffEntry(Integer.parseInt(parts[0].trim()), minLevel, maxLevel, mageOnly, fighterOnly));
			}
			catch (RuntimeException e)
			{
				LOGGER.warning(RLAgentEnvironmentConfig.class.getSimpleName() + ": Invalid buff entry: " + entry);
			}
		}
		return Collections.unmodifiableList(result);
	}
	
	private static List<FarmZone> parseFarmZones(String value)
	{
		final List<FarmZone> result = new ArrayList<>();
		for (String entry : value.split(";"))
		{
			final int colon = entry.indexOf(':');
			if (colon < 0)
			{
				continue;
			}
			try
			{
				final String[] range = entry.substring(0, colon).trim().split("-");
				final int minLevel = Integer.parseInt(range[0].trim());
				final int maxLevel = Integer.parseInt(range[range.length - 1].trim());
				final String anchorsValue = entry.substring(colon + 1).trim();
				if (anchorsValue.equalsIgnoreCase("start"))
				{
					result.add(new FarmZone(minLevel, maxLevel, List.of(), true));
					continue;
				}
				
				final List<Location> anchors = new ArrayList<>();
				for (String anchor : anchorsValue.split("\\|"))
				{
					final String[] xyz = anchor.trim().split(",");
					anchors.add(new Location(Integer.parseInt(xyz[0].trim()), Integer.parseInt(xyz[1].trim()), Integer.parseInt(xyz[2].trim())));
				}
				result.add(new FarmZone(minLevel, maxLevel, List.copyOf(anchors), false));
			}
			catch (RuntimeException e)
			{
				LOGGER.warning(RLAgentEnvironmentConfig.class.getSimpleName() + ": Invalid RLFarmZones entry: " + entry);
			}
		}
		return Collections.unmodifiableList(result);
	}
}
